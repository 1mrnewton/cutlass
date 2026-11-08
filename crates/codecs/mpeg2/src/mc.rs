//! Half-sample motion compensated prediction (H.262 7.6.4).
//!
//! A vector that reaches past the reference is not allowed in a conformant
//! stream (7.6.3.8). Such a vector still decodes: the samples outside the
//! picture repeat its edge, which is also what FFmpeg does.

use crate::frame::View;

/// Largest prediction block plus the half-sample neighbour.
const EDGE: usize = 17;

/// Where a prediction block is written.
pub(crate) struct Target<'a> {
    pub buf: &'a mut [u8],
    pub offset: usize,
    pub stride: usize,
}

/// Predicts a `size[0]`×`size[1]` block whose top-left sample in `src` is
/// `pos`, moved by the half-sample vector `mv`. With `average`, the result
/// is averaged into what the target holds (7.6.7).
pub(crate) fn predict(
    dst: &mut Target<'_>,
    src: View<'_>,
    pos: [i32; 2],
    mv: [i32; 2],
    size: [usize; 2],
    average: bool,
) {
    let half = [(mv[0] & 1) as usize, (mv[1] & 1) as usize];
    let sx = i64::from(pos[0]) + i64::from(mv[0] >> 1);
    let sy = i64::from(pos[1]) + i64::from(mv[1] >> 1);
    let inside = sx >= 0
        && sy >= 0
        && sx as u64 + (size[0] + half[0]) as u64 <= src.width as u64
        && sy as u64 + (size[1] + half[1]) as u64 <= src.height as u64;
    if inside {
        let base = src.offset + sy as usize * src.stride + sx as usize;
        kernel(dst, src.data, base, src.stride, size, half, average);
        return;
    }
    // Copy the needed area with its coordinates clamped to the picture.
    let mut edge = [0u8; EDGE * EDGE];
    let max_x = src.width as i64 - 1;
    let max_y = src.height as i64 - 1;
    for j in 0..size[1] + half[1] {
        let yy = (sy + j as i64).clamp(0, max_y) as usize;
        let row = src.offset + yy * src.stride;
        for i in 0..size[0] + half[0] {
            let xx = (sx + i as i64).clamp(0, max_x) as usize;
            edge[j * EDGE + i] = src.data[row + xx];
        }
    }
    kernel(dst, &edge, 0, EDGE, size, half, average);
}

#[inline(always)]
fn kernel(
    dst: &mut Target<'_>,
    src: &[u8],
    base: usize,
    stride: usize,
    size: [usize; 2],
    half: [usize; 2],
    average: bool,
) {
    if average {
        rows::<true>(dst, src, base, stride, size, half);
    } else {
        rows::<false>(dst, src, base, stride, size, half);
    }
}

/// The rows of one block. Every loop runs over slices of exactly `w`
/// samples so it compiles to vector code without bounds checks.
#[inline(always)]
fn rows<const AVERAGE: bool>(
    dst: &mut Target<'_>,
    src: &[u8],
    base: usize,
    stride: usize,
    size: [usize; 2],
    half: [usize; 2],
) {
    let [w, h] = size;
    for j in 0..h {
        let at = dst.offset + j * dst.stride;
        let d = &mut dst.buf[at..at + w];
        let row = base + j * stride;
        let a = &src[row..row + w];
        match half {
            [0, 0] if !AVERAGE => d.copy_from_slice(a),
            [0, 0] => {
                for (d, &x) in d.iter_mut().zip(a) {
                    *d = avg2(*d, x);
                }
            }
            [1, 0] | [0, 1] => {
                let next = if half[0] == 1 { row + 1 } else { row + stride };
                let b = &src[next..next + w];
                for ((d, &x), &y) in d.iter_mut().zip(a).zip(b) {
                    let p = avg2(x, y);
                    *d = if AVERAGE { avg2(*d, p) } else { p };
                }
            }
            _ => {
                let a1 = &src[row + 1..row + 1 + w];
                let b = &src[row + stride..row + stride + w];
                let b1 = &src[row + stride + 1..row + stride + 1 + w];
                for ((((d, &x), &x1), &y), &y1) in d.iter_mut().zip(a).zip(a1).zip(b).zip(b1) {
                    let sum = u16::from(x) + u16::from(x1) + u16::from(y) + u16::from(y1);
                    let p = ((sum + 2) >> 2) as u8;
                    *d = if AVERAGE { avg2(*d, p) } else { p };
                }
            }
        }
    }
}

#[inline(always)]
fn avg2(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b) + 1) >> 1) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(w: usize, h: usize) -> Vec<u8> {
        (0..w * h).map(|i| ((i * 37) % 251) as u8).collect()
    }

    fn view(data: &[u8], w: usize, h: usize) -> View<'_> {
        View {
            data,
            offset: 0,
            stride: w,
            width: w,
            height: h,
        }
    }

    /// The 7.6.4 formula sample by sample, with edge repeat.
    fn reference(src: &[u8], w: usize, h: usize, x: i32, y: i32, mv: [i32; 2]) -> i32 {
        let at = |xx: i32, yy: i32| {
            let xx = xx.clamp(0, w as i32 - 1) as usize;
            let yy = yy.clamp(0, h as i32 - 1) as usize;
            i32::from(src[yy * w + xx])
        };
        let (ix, iy) = (mv[0].div_euclid(2), mv[1].div_euclid(2));
        let (hx, hy) = (mv[0] - 2 * ix, mv[1] - 2 * iy);
        let (x, y) = (x + ix, y + iy);
        match (hx, hy) {
            (0, 0) => at(x, y),
            (0, 1) => (at(x, y) + at(x, y + 1) + 1) / 2,
            (1, 0) => (at(x, y) + at(x + 1, y) + 1) / 2,
            _ => (at(x, y) + at(x + 1, y) + at(x, y + 1) + at(x + 1, y + 1) + 2) / 4,
        }
    }

    #[test]
    fn every_half_sample_case_matches_the_formula_inside_and_past_the_edge() {
        let (w, h) = (32, 24);
        let src = plane(w, h);
        for mv in [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [-3, 5],
            [7, -9],
            [-41, -33],
            [60, 70],
        ] {
            for (x, y) in [(0, 0), (8, 8), (16, 8), (24, 16)] {
                let mut dst = [0u8; 64];
                let mut target = Target {
                    buf: &mut dst,
                    offset: 0,
                    stride: 8,
                };
                predict(&mut target, view(&src, w, h), [x, y], mv, [8, 8], false);
                for j in 0..8 {
                    for i in 0..8 {
                        let want = reference(&src, w, h, x + i as i32, y + j as i32, mv);
                        assert_eq!(i32::from(dst[j * 8 + i]), want, "mv {mv:?} at {x},{y}");
                    }
                }
            }
        }
    }

    #[test]
    fn averaging_rounds_up_the_half() {
        let src = [10u8; 64];
        let mut dst = [13u8; 64];
        let mut target = Target {
            buf: &mut dst,
            offset: 0,
            stride: 8,
        };
        predict(&mut target, view(&src, 8, 8), [0, 0], [0, 0], [8, 8], true);
        assert!(dst.iter().all(|&v| v == 12));
    }
}
