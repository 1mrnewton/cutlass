//! Sub-sample interpolation of inter prediction (clause 8.5.2.3): the
//! 8-tap filters across and then down, each pass rounded and held to the
//! sample range as libvpx does, and the average of two predictions.

use super::pixel::Pixel;

/// The filters of one kind, by sixteenth of a sample.
pub type Filter = [[i16; 8]; 16];

/// Predicts a `w`×`h` block into `dst` from `src`, whose element `at` is
/// the block's first reference sample. With a fraction across, `src` holds
/// 3 samples left and 4 right of each row; with one down, 3 rows above and
/// 4 below. With `avg`, the prediction is averaged into what `dst` holds.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn predict<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    src: &[P],
    at: usize,
    src_stride: usize,
    (w, h): (usize, usize),
    (mx, my): (usize, usize),
    filter: &Filter,
    max: i32,
    avg: bool,
) {
    let fx = (mx != 0).then(|| filter[mx]);
    let fy = (my != 0).then(|| filter[my]);
    if P::mc_fast(dst, dst_stride, src, at, src_stride, (w, h), (fx, fy), avg) {
        return;
    }
    match (mx != 0, my != 0) {
        (false, false) => {
            for r in 0..h {
                let s = &src[at + r * src_stride..at + r * src_stride + w];
                let d = &mut dst[r * dst_stride..r * dst_stride + w];
                if avg {
                    for (d, s) in d.iter_mut().zip(s) {
                        *d = P::new((d.get() + s.get() + 1) >> 1);
                    }
                } else {
                    d.copy_from_slice(s);
                }
            }
        }
        (true, false) => {
            let f = &filter[mx];
            for r in 0..h {
                let row = at + r * src_stride - 3;
                let d = &mut dst[r * dst_stride..r * dst_stride + w];
                horiz_row(d, &src[row..row + w + 7], f, max, avg);
            }
        }
        (false, true) => {
            let f = &filter[my];
            let base = at - 3 * src_stride;
            for r in 0..h {
                let d = &mut dst[r * dst_stride..r * dst_stride + w];
                vert_row(d, src, base + r * src_stride, src_stride, f, max, avg);
            }
        }
        (true, true) => {
            let fx = &filter[mx];
            let fy = &filter[my];
            let mut tmp = [P::default(); 64 * 71];
            let rows = h + 7;
            for r in 0..rows {
                let row = at + r * src_stride - 3 * src_stride - 3;
                horiz_row(
                    &mut tmp[r * 64..r * 64 + w],
                    &src[row..row + w + 7],
                    fx,
                    max,
                    false,
                );
            }
            for r in 0..h {
                let d = &mut dst[r * dst_stride..r * dst_stride + w];
                vert_row(d, &tmp, r * 64, 64, fy, max, avg);
            }
        }
    }
}

#[inline(always)]
fn horiz_row<P: Pixel>(d: &mut [P], s: &[P], f: &[i16; 8], max: i32, avg: bool) {
    for (x, out) in d.iter_mut().enumerate() {
        let t = &s[x..x + 8];
        let mut sum = 0i32;
        for k in 0..8 {
            sum += i32::from(f[k]) * t[k].get();
        }
        let v = ((sum + 64) >> 7).clamp(0, max);
        *out = if avg {
            P::new((out.get() + v + 1) >> 1)
        } else {
            P::new(v)
        };
    }
}

#[inline(always)]
fn vert_row<P: Pixel>(
    d: &mut [P],
    s: &[P],
    base: usize,
    stride: usize,
    f: &[i16; 8],
    max: i32,
    avg: bool,
) {
    for (x, out) in d.iter_mut().enumerate() {
        let mut sum = 0i32;
        for k in 0..8 {
            sum += i32::from(f[k]) * s[base + k * stride + x].get();
        }
        let v = ((sum + 64) >> 7).clamp(0, max);
        *out = if avg {
            P::new((out.get() + v + 1) >> 1)
        } else {
            P::new(v)
        };
    }
}

/// Prediction from a reference of another size (libvpx `vpx_scaled_2d`):
/// sample `(x, y)` of the reference is `get(x, y)`, held to the plane by
/// the caller. `x0`, `y0` are the first full sample and `sx`, `sy` the
/// sixteenths, stepped by `xs`, `ys` sixteenths a sample.
#[allow(clippy::too_many_arguments)]
pub fn predict_scaled<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    get: impl Fn(i32, i32) -> P,
    (w, h): (usize, usize),
    (x0, y0): (i32, i32),
    (sx, sy): (i32, i32),
    (xs, ys): (i32, i32),
    filter: &Filter,
    max: i32,
    avg: bool,
) {
    let tap = |s: [i32; 8], f: &[i16; 8]| {
        let mut sum = 0;
        for k in 0..8 {
            sum += i32::from(f[k]) * s[k];
        }
        ((sum + 64) >> 7).clamp(0, max)
    };
    let horiz = xs != 16 || sx != 0;
    let vert = ys != 16 || sy != 0;
    // Rows of the horizontally filtered window.
    let rows = if vert {
        ((((h as i32 - 1) * ys + sy) >> 4) + 8) as usize
    } else {
        h
    };
    let top = if vert { y0 - 3 } else { y0 };
    let mut tmp = vec![0i32; rows * w];
    for r in 0..rows {
        let y = top + r as i32;
        let mut xq = sx;
        for c in 0..w {
            tmp[r * w + c] = if horiz {
                let x = x0 + (xq >> 4);
                let f = &filter[(xq & 15) as usize];
                tap(core::array::from_fn(|k| get(x - 3 + k as i32, y).get()), f)
            } else {
                get(x0 + c as i32, y).get()
            };
            xq += xs;
        }
    }
    for r in 0..h {
        for c in 0..w {
            let v = if vert {
                let yq = sy + r as i32 * ys;
                let base = (yq >> 4) as usize;
                let f = &filter[(yq & 15) as usize];
                tap(core::array::from_fn(|k| tmp[(base + k) * w + c]), f)
            } else {
                tmp[r * w + c]
            };
            let d = &mut dst[r * dst_stride + c];
            *d = if avg {
                P::new((d.get() + v + 1) >> 1)
            } else {
                P::new(v)
            };
        }
    }
}
