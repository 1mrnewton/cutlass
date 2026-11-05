//! Motion-compensated prediction (RFC 6386 Section 18).
//!
//! A reference plane extends past its edges by repeating its outermost
//! pixels, as far as any motion vector reaches. The plane's size is the
//! macroblock-aligned size the decoder reconstructs.

use crate::tables::{BILINEAR, SIXTAP};

/// One reference plane.
#[derive(Clone, Copy)]
pub struct RefPlane<'a> {
    pub data: &'a [u8],
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

/// Interpolation filters by bitstream version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    SixTap,
    Bilinear,
}

impl Filter {
    fn taps(self, frac: usize) -> &'static [i32; 6] {
        match self {
            Self::SixTap => &SIXTAP[frac & 7],
            Self::Bilinear => &BILINEAR[frac & 7],
        }
    }
}

/// Largest block predicted at once.
const MAX: usize = 16;
/// Rows and columns a six-tap filter reads around a block.
const PAD: usize = 5;

/// Predicts a `w`×`h` block whose top-left pixel sits at `(x, y)` in
/// whole pixels plus `fx`/`fy` eighths, writing it to `out` with stride
/// `out_stride`.
#[allow(clippy::too_many_arguments)]
pub fn predict(
    plane: RefPlane<'_>,
    x: i32,
    y: i32,
    fx: usize,
    fy: usize,
    w: usize,
    h: usize,
    filter: Filter,
    out: &mut [u8],
    out_stride: usize,
) {
    // Source window: two pixels before, three after, in each direction.
    let mut window = [0u8; (MAX + PAD) * (MAX + PAD)];
    let ww = w + PAD;
    let wh = h + PAD;
    let x0 = x - 2;
    let y0 = y - 2;
    let inside = x0 >= 0
        && y0 >= 0
        && (x0 as usize) + ww <= plane.width
        && (y0 as usize) + wh <= plane.height;
    let (src, src_stride, base): (&[u8], usize, usize) = if inside {
        (
            plane.data,
            plane.stride,
            (y0 as usize) * plane.stride + x0 as usize,
        )
    } else {
        let max_x = plane.width as i32 - 1;
        let max_y = plane.height as i32 - 1;
        for r in 0..wh {
            let sy = (y0 + r as i32).clamp(0, max_y) as usize;
            let row = &plane.data[sy * plane.stride..sy * plane.stride + plane.width];
            for c in 0..ww {
                let sx = (x0 + c as i32).clamp(0, max_x) as usize;
                window[r * ww + c] = row[sx];
            }
        }
        (&window[..], ww, 0)
    };

    if fx == 0 && fy == 0 {
        for r in 0..h {
            let s = base + (r + 2) * src_stride + 2;
            out[r * out_stride..r * out_stride + w].copy_from_slice(&src[s..s + w]);
        }
        return;
    }
    if fy == 0 {
        let taps = filter.taps(fx);
        for r in 0..h {
            let s = base + (r + 2) * src_stride;
            horizontal(
                &src[s..s + w + PAD],
                taps,
                &mut out[r * out_stride..r * out_stride + w],
            );
        }
        return;
    }
    let vtaps = filter.taps(fy);
    if fx == 0 {
        for r in 0..h {
            let s = base + r * src_stride + 2;
            vertical(
                src,
                s,
                src_stride,
                vtaps,
                &mut out[r * out_stride..r * out_stride + w],
            );
        }
        return;
    }
    let htaps = filter.taps(fx);
    let mut mid = [0u8; MAX * (MAX + PAD)];
    for r in 0..wh {
        let s = base + r * src_stride;
        horizontal(&src[s..s + w + PAD], htaps, &mut mid[r * w..r * w + w]);
    }
    for r in 0..h {
        vertical(
            &mid,
            r * w,
            w,
            vtaps,
            &mut out[r * out_stride..r * out_stride + w],
        );
    }
}

/// One row through the horizontal filter: `src` holds `out.len() + 5`
/// pixels starting two before the first output.
#[inline]
fn horizontal(src: &[u8], taps: &[i32; 6], out: &mut [u8]) {
    let w = out.len();
    let s0 = &src[..w];
    let s1 = &src[1..=w];
    let s2 = &src[2..w + 2];
    let s3 = &src[3..w + 3];
    let s4 = &src[4..w + 4];
    let s5 = &src[5..w + 5];
    for c in 0..w {
        let sum = i32::from(s0[c]) * taps[0]
            + i32::from(s1[c]) * taps[1]
            + i32::from(s2[c]) * taps[2]
            + i32::from(s3[c]) * taps[3]
            + i32::from(s4[c]) * taps[4]
            + i32::from(s5[c]) * taps[5];
        out[c] = ((sum + 64) >> 7).clamp(0, 255) as u8;
    }
}

/// One row through the vertical filter: the six input rows start at `at`
/// (two rows above the output row) with `stride` between them.
#[inline]
fn vertical(src: &[u8], at: usize, stride: usize, taps: &[i32; 6], out: &mut [u8]) {
    let w = out.len();
    let row = |k: usize| &src[at + k * stride..at + k * stride + w];
    let (r0, r1, r2, r3, r4, r5) = (row(0), row(1), row(2), row(3), row(4), row(5));
    for c in 0..w {
        let sum = i32::from(r0[c]) * taps[0]
            + i32::from(r1[c]) * taps[1]
            + i32::from(r2[c]) * taps[2]
            + i32::from(r3[c]) * taps[3]
            + i32::from(r4[c]) * taps[4]
            + i32::from(r5[c]) * taps[5];
        out[c] = ((sum + 64) >> 7).clamp(0, 255) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::{Filter, RefPlane, predict};

    fn plane(w: usize, h: usize) -> Vec<u8> {
        (0..w * h)
            .map(|i| ((i * 37 + i / w * 11) % 251) as u8)
            .collect()
    }

    /// The reference decoder's way: always two passes over an explicitly
    /// edge-extended window.
    #[allow(clippy::too_many_arguments)]
    fn slow(
        p: &[u8],
        w: usize,
        h: usize,
        x: i32,
        y: i32,
        fx: usize,
        fy: usize,
        bw: usize,
        bh: usize,
    ) -> Vec<u8> {
        let at = |xx: i32, yy: i32| -> i32 {
            let cx = xx.clamp(0, w as i32 - 1) as usize;
            let cy = yy.clamp(0, h as i32 - 1) as usize;
            i32::from(p[cy * w + cx])
        };
        let ht = crate::tables::SIXTAP[fx];
        let vt = crate::tables::SIXTAP[fy];
        let mut mid = vec![0i32; bw * (bh + 5)];
        for r in 0..bh + 5 {
            for c in 0..bw {
                let mut sum = 64;
                for (t, &tap) in ht.iter().enumerate() {
                    sum += at(x + c as i32 + t as i32 - 2, y + r as i32 - 2) * tap;
                }
                mid[r * bw + c] = (sum >> 7).clamp(0, 255);
            }
        }
        let mut out = vec![0u8; bw * bh];
        for r in 0..bh {
            for c in 0..bw {
                let mut sum = 64;
                for t in 0..6 {
                    sum += mid[(r + t) * bw + c] * vt[t];
                }
                out[r * bw + c] = (sum >> 7).clamp(0, 255) as u8;
            }
        }
        out
    }

    #[test]
    fn every_shortcut_matches_two_full_passes() {
        let (w, h) = (32, 32);
        let p = plane(w, h);
        let rp = RefPlane {
            data: &p,
            stride: w,
            width: w,
            height: h,
        };
        for (x, y) in [(8, 8), (-30, 4), (40, -9), (28, 30), (-3, -3), (1, 29)] {
            for fx in 0..8 {
                for fy in 0..8 {
                    for (bw, bh) in [(16, 16), (8, 8), (4, 4)] {
                        let mut out = vec![0u8; bw * bh];
                        predict(rp, x, y, fx, fy, bw, bh, Filter::SixTap, &mut out, bw);
                        assert_eq!(
                            out,
                            slow(&p, w, h, x, y, fx, fy, bw, bh),
                            "{x},{y} {fx},{fy} {bw}"
                        );
                    }
                }
            }
        }
    }
}
