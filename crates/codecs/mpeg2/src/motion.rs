//! Motion estimation for the encoder: one 16×16 frame vector per
//! macroblock, in half samples.
//!
//! The search starts from the vectors of neighbouring macroblocks and zero,
//! walks a diamond at four, two, and one sample, then tries the eight half
//! sample neighbours. Every vector keeps its block inside the coded picture
//! (7.6.3.8), so any decoder predicts the same samples.

use crate::frame::{Frame, Geometry, View};
use crate::mc::{Target, predict};

/// Largest vector component searched, in half samples. `f_code` 4 codes
/// -128 to 127, inside every level's vertical limit.
pub(crate) const SEARCH_LIMIT: i32 = 126;

/// A chosen vector, its sum of absolute differences, and that plus the
/// weighted vector bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Found {
    pub(crate) mv: [i32; 2],
    pub(crate) sad: u32,
    pub(crate) cost: u32,
}

/// Bits of one vector component difference, near enough for decisions.
pub(crate) fn mv_bits(delta: i32) -> u32 {
    if delta == 0 {
        1
    } else {
        2 + 2 * (32 - delta.unsigned_abs().leading_zeros())
    }
}

/// Searches one reference for every macroblock of a picture.
pub(crate) struct Search<'a> {
    geo: &'a Geometry,
    src: &'a Frame,
    reference: &'a Frame,
    /// Weight of one vector bit against one unit of SAD.
    pub(crate) lambda: u32,
}

impl<'a> Search<'a> {
    pub(crate) fn new(
        geo: &'a Geometry,
        src: &'a Frame,
        reference: &'a Frame,
        lambda: u32,
    ) -> Self {
        Self {
            geo,
            src,
            reference,
            lambda,
        }
    }

    /// Allowed half-sample range per component for the block at `mbx, mby`.
    fn bounds(&self, mbx: usize, mby: usize) -> [[i32; 2]; 2] {
        let (stride, height) = self.geo.plane(0);
        let x2 = (mbx * 32) as i32;
        let y2 = (mby * 32) as i32;
        let max_x = 2 * (stride as i32 - 16) - x2;
        let max_y = 2 * (height as i32 - 16) - y2;
        [
            [(-x2).max(-SEARCH_LIMIT), max_x.min(SEARCH_LIMIT)],
            [(-y2).max(-SEARCH_LIMIT), max_y.min(SEARCH_LIMIT)],
        ]
    }

    fn clamp(bounds: &[[i32; 2]; 2], mv: [i32; 2]) -> [i32; 2] {
        [
            mv[0].clamp(bounds[0][0], bounds[0][1]),
            mv[1].clamp(bounds[1][0], bounds[1][1]),
        ]
    }

    /// SAD at a full-sample vector (both components even).
    fn sad_full(&self, mbx: usize, mby: usize, mv: [i32; 2]) -> u32 {
        let (stride, _) = self.geo.plane(0);
        let src = &self.src.planes[0];
        let reference = &self.reference.planes[0];
        let sx = mbx * 16;
        let sy = mby * 16;
        let rx = (sx as i32 + mv[0] / 2) as usize;
        let ry = (sy as i32 + mv[1] / 2) as usize;
        let mut total = 0u32;
        for row in 0..16 {
            let a = &src[(sy + row) * stride + sx..][..16];
            let b = &reference[(ry + row) * stride + rx..][..16];
            total += sad16(a, b);
        }
        total
    }

    /// SAD at any half-sample vector.
    fn sad_half(&self, mbx: usize, mby: usize, mv: [i32; 2]) -> u32 {
        if mv[0] & 1 == 0 && mv[1] & 1 == 0 {
            return self.sad_full(mbx, mby, mv);
        }
        let mut block = [0u8; 256];
        let mut dst = Target {
            buf: &mut block,
            offset: 0,
            stride: 16,
        };
        let view = View::frame(self.reference, self.geo, 0);
        predict(
            &mut dst,
            view,
            [(mbx * 16) as i32, (mby * 16) as i32],
            mv,
            [16, 16],
            false,
        );
        let (stride, _) = self.geo.plane(0);
        let src = &self.src.planes[0];
        let mut total = 0u32;
        for row in 0..16 {
            let a = &src[(mby * 16 + row) * stride + mbx * 16..][..16];
            total += sad16(a, &block[row * 16..row * 16 + 16]);
        }
        total
    }

    fn cost(&self, sad: u32, mv: [i32; 2], pred: [i32; 2]) -> u32 {
        sad + self.lambda * (mv_bits(mv[0] - pred[0]) + mv_bits(mv[1] - pred[1]))
    }

    /// Best vector for one macroblock. `pred` is the vector the bitstream
    /// predicts from; `candidates` seed the search.
    pub(crate) fn block(
        &self,
        mbx: usize,
        mby: usize,
        pred: [i32; 2],
        candidates: &[[i32; 2]],
    ) -> Found {
        let bounds = self.bounds(mbx, mby);
        let even = |v: i32| v & !1;
        let mut best_mv = [0, 0];
        let mut best_cost = u32::MAX;
        let mut best_sad = u32::MAX;
        let consider =
            |mv: [i32; 2], best_mv: &mut [i32; 2], best_cost: &mut u32, best_sad: &mut u32| {
                let sad = self.sad_full(mbx, mby, mv);
                let cost = self.cost(sad, mv, pred);
                if cost < *best_cost {
                    *best_cost = cost;
                    *best_sad = sad;
                    *best_mv = mv;
                }
            };
        let zero = Self::clamp(&bounds, [0, 0]);
        consider(
            [even(zero[0]), even(zero[1])],
            &mut best_mv,
            &mut best_cost,
            &mut best_sad,
        );
        for &candidate in candidates.iter().chain(std::iter::once(&pred)) {
            let mv = Self::clamp(&bounds, [even(candidate[0]), even(candidate[1])]);
            let mv = [even(mv[0]), even(mv[1])];
            if mv != best_mv {
                consider(mv, &mut best_mv, &mut best_cost, &mut best_sad);
            }
        }
        // Diamond at 4, 2, then 1 sample.
        for step in [8, 4, 2] {
            for _ in 0..32 {
                let centre = best_mv;
                for (dx, dy) in [(step, 0), (-step, 0), (0, step), (0, -step)] {
                    let mv = [centre[0] + dx, centre[1] + dy];
                    if mv[0] < bounds[0][0]
                        || mv[0] > bounds[0][1]
                        || mv[1] < bounds[1][0]
                        || mv[1] > bounds[1][1]
                    {
                        continue;
                    }
                    consider(mv, &mut best_mv, &mut best_cost, &mut best_sad);
                }
                if best_mv == centre {
                    break;
                }
            }
        }
        // Half samples around the best full sample.
        let centre = best_mv;
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let mv = [centre[0] + dx, centre[1] + dy];
                if mv[0] < bounds[0][0]
                    || mv[0] > bounds[0][1]
                    || mv[1] < bounds[1][0]
                    || mv[1] > bounds[1][1]
                {
                    continue;
                }
                let sad = self.sad_half(mbx, mby, mv);
                let cost = self.cost(sad, mv, pred);
                if cost < best_cost {
                    best_cost = cost;
                    best_sad = sad;
                    best_mv = mv;
                }
            }
        }
        Found {
            mv: best_mv,
            sad: best_sad,
            cost: best_cost,
        }
    }

    /// The SAD and cost of one given vector, clamped into the picture.
    pub(crate) fn evaluate(&self, mbx: usize, mby: usize, pred: [i32; 2], mv: [i32; 2]) -> Found {
        let mv = Self::clamp(&self.bounds(mbx, mby), mv);
        let sad = self.sad_half(mbx, mby, mv);
        Found {
            mv,
            sad,
            cost: self.cost(sad, mv, pred),
        }
    }
}

/// SAD of the averaged forward and backward luma prediction of one
/// macroblock, vectors `mv[s]` already inside the picture.
pub(crate) fn sad_bi(
    geo: &Geometry,
    src: &Frame,
    refs: [&Frame; 2],
    mbx: usize,
    mby: usize,
    mv: [[i32; 2]; 2],
) -> u32 {
    let mut block = [0u8; 256];
    for (s, reference) in refs.into_iter().enumerate() {
        let mut dst = Target {
            buf: &mut block,
            offset: 0,
            stride: 16,
        };
        predict(
            &mut dst,
            View::frame(reference, geo, 0),
            [(mbx * 16) as i32, (mby * 16) as i32],
            mv[s],
            [16, 16],
            s == 1,
        );
    }
    let (stride, _) = geo.plane(0);
    let mut total = 0u32;
    for row in 0..16 {
        let a = &src.planes[0][(mby * 16 + row) * stride + mbx * 16..][..16];
        total += sad16(a, &block[row * 16..row * 16 + 16]);
    }
    total
}

#[inline]
fn sad16(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| u32::from(x.abs_diff(y)))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headers::CHROMA_420;

    fn geo() -> Geometry {
        Geometry {
            mb_width: 6,
            mb_height: 4,
            chroma_format: CHROMA_420,
        }
    }

    /// A smooth texture with detail at every scale, so SAD has one minimum.
    fn texture(geo: &Geometry, dx: i32, dy: i32) -> Frame {
        let mut frame = Frame::new(geo, 0);
        let (stride, height) = geo.plane(0);
        for y in 0..height {
            for x in 0..stride {
                let (u, v) = (x as i32 + dx, y as i32 + dy);
                let value = 128.0
                    + 60.0 * ((u as f64) * 0.21).sin() * ((v as f64) * 0.17).cos()
                    + 30.0 * ((u as f64 * 0.05 + v as f64 * 0.09).sin());
                frame.planes[0][y * stride + x] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
        frame
    }

    #[test]
    fn finds_a_full_sample_shift() {
        let geo = geo();
        let reference = texture(&geo, 0, 0);
        let src = texture(&geo, 3, -2);
        let search = Search::new(&geo, &src, &reference, 0);
        let found = search.block(2, 1, [0, 0], &[]);
        assert_eq!(found.mv, [6, -4]);
        assert_eq!(found.sad, 0);
    }

    #[test]
    fn vectors_stay_inside_the_picture() {
        let geo = geo();
        let reference = texture(&geo, 0, 0);
        let src = texture(&geo, -20, -20);
        let search = Search::new(&geo, &src, &reference, 0);
        for (mbx, mby) in [(0, 0), (5, 3), (0, 3), (5, 0)] {
            let found = search.block(mbx, mby, [0, 0], &[[-200, 300]]);
            let x2 = (mbx * 32) as i32 + found.mv[0];
            let y2 = (mby * 32) as i32 + found.mv[1];
            assert!((0..=2 * (96 - 16)).contains(&x2), "{mbx},{mby}: {found:?}");
            assert!((0..=2 * (64 - 16)).contains(&y2), "{mbx},{mby}: {found:?}");
        }
    }

    #[test]
    fn half_samples_refine_the_vector() {
        let geo = geo();
        let reference = texture(&geo, 0, 0);
        // The source is the reference averaged with its right neighbour:
        // exactly the half-sample prediction at (+1, 0).
        let mut src = reference.clone();
        let (stride, height) = geo.plane(0);
        for y in 0..height {
            for x in 0..stride - 1 {
                let a = u32::from(reference.planes[0][y * stride + x]);
                let b = u32::from(reference.planes[0][y * stride + x + 1]);
                src.planes[0][y * stride + x] = (a + b).div_ceil(2) as u8;
            }
        }
        let search = Search::new(&geo, &src, &reference, 0);
        let found = search.block(2, 1, [0, 0], &[]);
        assert_eq!(found.mv, [1, 0]);
        assert_eq!(found.sad, 0);
    }

    #[test]
    fn vector_bits_grow_with_the_difference() {
        assert_eq!(mv_bits(0), 1);
        assert!(mv_bits(1) < mv_bits(16));
        assert_eq!(mv_bits(-5), mv_bits(5));
    }
}
