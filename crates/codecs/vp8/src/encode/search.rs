//! Motion search on the luma plane.

use crate::inter::{self, Filter, RefPlane};
use crate::modes::Mv;

/// Sum of absolute differences of a 16×16 block of the reference at whole
/// pixel `(x, y)` against `src` (stride 16).
#[must_use]
pub fn sad16(plane: RefPlane<'_>, x: i32, y: i32, src: &[u8; 256], limit: u32) -> u32 {
    let inside = x >= 0
        && y >= 0
        && (x as usize) + 16 <= plane.width
        && (y as usize) + 16 <= plane.height;
    let mut total = 0u32;
    if inside {
        let base = y as usize * plane.stride + x as usize;
        for r in 0..16 {
            let row = &plane.data[base + r * plane.stride..base + r * plane.stride + 16];
            let s = &src[r * 16..r * 16 + 16];
            total += row
                .iter()
                .zip(s)
                .map(|(&a, &b)| u32::from(a.abs_diff(b)))
                .sum::<u32>();
            if total >= limit {
                return total;
            }
        }
        return total;
    }
    let mut block = [0u8; 256];
    inter::predict(plane, x, y, 0, 0, 16, 16, Filter::SixTap, &mut block, 16);
    block
        .iter()
        .zip(src.iter())
        .map(|(&a, &b)| u32::from(a.abs_diff(b)))
        .sum()
}

/// Sum of squared differences between two 16×16 blocks.
#[must_use]
pub fn sse16(a: &[u8; 256], b: &[u8; 256]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = i32::from(x) - i32::from(y);
            (d * d) as u32
        })
        .sum()
}

/// Predicts the 16×16 luma block at macroblock `(mbx, mby)` moved by `mv`.
pub fn predict16(plane: RefPlane<'_>, mbx: usize, mby: usize, mv: Mv, filter: Filter, out: &mut [u8; 256]) {
    inter::predict(
        plane,
        (mbx * 16) as i32 + (mv.col >> 2),
        (mby * 16) as i32 + (mv.row >> 2),
        ((mv.col & 3) * 2) as usize,
        ((mv.row & 3) * 2) as usize,
        16,
        16,
        filter,
        out,
        16,
    );
}

/// The area a search may cover, in quarter pixels.
#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub min: Mv,
    pub max: Mv,
}

impl Window {
    fn contains(&self, mv: Mv) -> bool {
        mv.row >= self.min.row && mv.row <= self.max.row && mv.col >= self.min.col && mv.col <= self.max.col
    }
}

/// A search request for one macroblock.
pub struct Search<'a> {
    pub plane: RefPlane<'a>,
    pub src: &'a [u8; 256],
    pub mbx: usize,
    pub mby: usize,
    pub window: Window,
    /// Penalty in SAD units for a vector, from its coding cost.
    pub penalty: &'a dyn Fn(Mv) -> u32,
    pub filter: Filter,
    /// Integer search steps to try; more is slower and finds more.
    pub range: i32,
    /// Whether to refine to quarter pixels.
    pub subpel: bool,
}

/// Diamond search from the best of `starts`, then half and quarter pixel
/// refinement. Returns the vector and its SAD-plus-penalty score.
#[must_use]
pub fn search(s: &Search<'_>, starts: &[Mv]) -> (Mv, u32) {
    let x0 = (s.mbx * 16) as i32;
    let y0 = (s.mby * 16) as i32;
    let full = |mv: Mv, limit: u32| -> u32 {
        let p = (s.penalty)(mv);
        if p >= limit {
            return u32::MAX;
        }
        sad16(s.plane, x0 + (mv.col >> 2), y0 + (mv.row >> 2), s.src, limit - p).saturating_add(p)
    };
    let mut best = Mv::ZERO;
    let mut best_score = u32::MAX;
    for &start in starts {
        // Round to whole pixels.
        let mv = Mv::new((start.row + 2) & !3, (start.col + 2) & !3);
        if !s.window.contains(mv) {
            continue;
        }
        let score = full(mv, best_score);
        if score < best_score {
            best = mv;
            best_score = score;
        }
    }
    if best_score == u32::MAX {
        best = Mv::ZERO;
        best_score = full(best, u32::MAX);
    }
    // Large diamond with a shrinking step, in whole pixels.
    let mut step = s.range.max(1);
    while step >= 1 {
        let mut moved = true;
        while moved {
            moved = false;
            let center = best;
            for (dr, dc) in [(-step, 0), (step, 0), (0, -step), (0, step)] {
                let mv = Mv::new(center.row + dr * 4, center.col + dc * 4);
                if !s.window.contains(mv) {
                    continue;
                }
                let score = full(mv, best_score);
                if score < best_score {
                    best = mv;
                    best_score = score;
                    moved = true;
                }
            }
        }
        step /= 2;
    }
    for (dr, dc) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
        let mv = Mv::new(best.row + dr * 4, best.col + dc * 4);
        if s.window.contains(mv) {
            let score = full(mv, best_score);
            if score < best_score {
                best = mv;
                best_score = score;
            }
        }
    }
    if !s.subpel {
        return (best, best_score);
    }
    // Sub-pixel refinement with the real interpolation filter.
    let mut block = [0u8; 256];
    let sub = |mv: Mv, block: &mut [u8; 256]| -> u32 {
        predict16(s.plane, s.mbx, s.mby, mv, s.filter, block);
        let sad: u32 = block
            .iter()
            .zip(s.src.iter())
            .map(|(&a, &b)| u32::from(a.abs_diff(b)))
            .sum();
        sad + (s.penalty)(mv)
    };
    for step in [2, 1] {
        let center = best;
        for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] {
            let mv = Mv::new(center.row + dr * step, center.col + dc * step);
            if !s.window.contains(mv) {
                continue;
            }
            let score = sub(mv, &mut block);
            if score < best_score {
                best = mv;
                best_score = score;
            }
        }
    }
    (best, best_score)
}

#[cfg(test)]
mod tests {
    use super::{Search, Window, search};
    use crate::inter::{Filter, RefPlane};
    use crate::modes::Mv;

    #[test]
    fn search_finds_a_shifted_block() {
        let (w, h) = (64usize, 64usize);
        let plane: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                (128.0 + 60.0 * (x * 0.3).sin() + 50.0 * (y * 0.23).cos()) as u8
            })
            .collect();
        let rp = RefPlane {
            data: &plane,
            stride: w,
            width: w,
            height: h,
        };
        // The source is the reference at (16 + 5, 16 - 3).
        let mut src = [0u8; 256];
        for r in 0..16 {
            for c in 0..16 {
                src[r * 16 + c] = plane[(13 + r) * w + 21 + c];
            }
        }
        let penalty = |_: Mv| 0;
        let s = Search {
            plane: rp,
            src: &src,
            mbx: 1,
            mby: 1,
            window: Window {
                min: Mv::new(-64, -64),
                max: Mv::new(64, 64),
            },
            penalty: &penalty,
            filter: Filter::SixTap,
            range: 8,
            subpel: true,
        };
        let (mv, score) = search(&s, &[Mv::ZERO]);
        assert_eq!(mv, Mv::new(-12, 20));
        assert_eq!(score, 0);
    }
}
