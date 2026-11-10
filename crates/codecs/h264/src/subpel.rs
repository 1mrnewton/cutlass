//! Half-sample luma planes of one reference picture, for the encoder's motion search.
//!
//! Clause 8.4.2.2.1 builds every quarter sample from the full sample `G`, the
//! horizontal half `b`, the vertical half `h` and the centre `j`, or the average
//! of two of them. Those four depend only on the reference, so they are filtered
//! once per picture here, with a margin of edge samples, instead of again for
//! every candidate vector. A block whose samples all fall inside the margin is
//! then a plain read, or the average of two reads, per sample.
//!
//! Every value equals [`crate::recon::qpel`] at the same position, edges
//! included: each plane is computed with the same clamped taps.

use crate::recon;

/// Edge samples kept on each side. The search reaches 16 samples past a block,
/// and the quarter positions read one more to the right and below.
const MARGIN: i32 = 24;

/// The widest block the search compares.
pub const MAX_BLOCK: usize = 16;

pub struct Subpel {
    stride: usize,
    width: i32,
    height: i32,
    /// `G`, the full samples.
    full: Vec<u8>,
    /// `b`, the horizontal half samples.
    half_h: Vec<u8>,
    /// `h`, the vertical half samples.
    half_v: Vec<u8>,
    /// `j`, the centre half samples.
    centre: Vec<u8>,
}

#[derive(Clone, Copy)]
enum Plane {
    Full,
    HalfH,
    HalfV,
    Centre,
}

/// One plane read at an offset of 0 or 1 sample right and below.
type Read = (Plane, i32, i32);

impl Subpel {
    pub fn new(plane: &[u8], width: u32, height: u32) -> Self {
        let w = width as i32;
        let h = height as i32;
        let stride = (w + 2 * MARGIN) as usize;
        let rows = (h + 2 * MARGIN) as usize;
        let src_stride = width as usize;
        // The centre needs the raw horizontal taps three rows below and two above.
        let raw_rows = rows + 5;
        let edge = |x: i32, y: i32, horizontal: bool| {
            recon::tap(plane, src_stride, w, h, x, y, horizontal)
        };
        // Columns whose six horizontal taps all fall inside the picture, as
        // offsets into a margined row. The rest clamp through `recon::tap`.
        let inner_x = (MARGIN + 2) as usize..(MARGIN + w - 3).max(MARGIN + 2) as usize;
        let mut raw = vec![0i32; stride * raw_rows];
        for (ry, line) in raw.chunks_exact_mut(stride).enumerate() {
            let y = ry as i32 - MARGIN - 2;
            let yc = y.clamp(0, h - 1) as usize;
            let src = &plane[yc * src_stride..(yc + 1) * src_stride];
            for (rx, value) in line.iter_mut().enumerate() {
                if !inner_x.contains(&rx) {
                    *value = edge(rx as i32 - MARGIN, y, true);
                }
            }
            if inner_x.is_empty() {
                continue;
            }
            let start = inner_x.start;
            for (i, value) in line[inner_x.clone()].iter_mut().enumerate() {
                let s = &src[start + i - MARGIN as usize - 2..start + i - MARGIN as usize + 4];
                *value = six(s[0], s[1], s[2], s[3], s[4], s[5]);
            }
        }
        let mut full = vec![0u8; stride * rows];
        let mut half_h = vec![0u8; stride * rows];
        let mut half_v = vec![0u8; stride * rows];
        let mut centre = vec![0u8; stride * rows];
        for row in 0..rows {
            let y = row as i32 - MARGIN;
            let line = row * stride..(row + 1) * stride;
            let yc = y.clamp(0, h - 1) as usize;
            let src = &plane[yc * src_stride..(yc + 1) * src_stride];
            for (col, value) in full[line.clone()].iter_mut().enumerate() {
                *value = src[(col as i32 - MARGIN).clamp(0, w - 1) as usize];
            }
            let taps = &raw[(row + 2) * stride..(row + 3) * stride];
            for (value, &tap) in half_h[line.clone()].iter_mut().zip(taps) {
                *value = clip((tap + 16) >> 5);
            }
            let out = &mut half_v[line.clone()];
            if (2..h - 3).contains(&y) {
                // Six whole rows: only the columns past the picture's sides clamp.
                let at = |k: i32| {
                    let r = (y + k) as usize;
                    &plane[r * src_stride..(r + 1) * src_stride]
                };
                let rows6 = [at(-2), at(-1), at(0), at(1), at(2), at(3)];
                for (col, value) in out.iter_mut().enumerate() {
                    let x = (col as i32 - MARGIN).clamp(0, w - 1) as usize;
                    let tap = six(
                        rows6[0][x],
                        rows6[1][x],
                        rows6[2][x],
                        rows6[3][x],
                        rows6[4][x],
                        rows6[5][x],
                    );
                    *value = clip((tap + 16) >> 5);
                }
            } else {
                for (col, value) in out.iter_mut().enumerate() {
                    *value = clip((edge(col as i32 - MARGIN, y, false) + 16) >> 5);
                }
            }
            let r = |k: usize| &raw[(row + k) * stride..(row + k + 1) * stride];
            let (r0, r1, r2, r3, r4, r5) = (r(0), r(1), r(2), r(3), r(4), r(5));
            for (col, value) in centre[line].iter_mut().enumerate() {
                let acc =
                    r0[col] - 5 * r1[col] + 20 * r2[col] + 20 * r3[col] - 5 * r4[col] + r5[col];
                *value = clip((acc + 512) >> 10);
            }
        }
        Self {
            stride,
            width: w,
            height: h,
            full,
            half_h,
            half_v,
            centre,
        }
    }

    /// The `w`×`h` prediction at (`x`, `y`) moved by the quarter-sample `mv`,
    /// row by row into `out` with a stride of [`MAX_BLOCK`]. `None` when the
    /// block or a sample it reads falls outside the picture and its margin, so
    /// the caller takes the clamped path.
    pub fn block(
        &self,
        (x, y, w, h): (u32, u32, u32, u32),
        mv: [i16; 2],
        out: &mut [u8; MAX_BLOCK * MAX_BLOCK],
    ) -> Option<()> {
        if w as usize > MAX_BLOCK || h as usize > MAX_BLOCK {
            return None;
        }
        let mvx = i32::from(mv[0]);
        let mvy = i32::from(mv[1]);
        let left = x as i32 + (mvx >> 2);
        let top = y as i32 + (mvy >> 2);
        // One extra column and row for the reads offset right and below.
        if left < -MARGIN
            || top < -MARGIN
            || left + w as i32 + 1 > self.width + MARGIN
            || top + h as i32 + 1 > self.height + MARGIN
        {
            return None;
        }
        let (first, second) = reads(mvx & 3, mvy & 3);
        let w = w as usize;
        for row in 0..h as usize {
            let line = &mut out[row * MAX_BLOCK..row * MAX_BLOCK + w];
            let a = self.row(first, left, top + row as i32, w);
            match second {
                None => line.copy_from_slice(a),
                Some(second) => {
                    let b = self.row(second, left, top + row as i32, w);
                    for ((dst, &p), &q) in line.iter_mut().zip(a).zip(b) {
                        *dst = ((u16::from(p) + u16::from(q) + 1) >> 1) as u8;
                    }
                }
            }
        }
        Some(())
    }

    fn row(&self, (plane, dx, dy): Read, x: i32, y: i32, w: usize) -> &[u8] {
        let data = match plane {
            Plane::Full => &self.full,
            Plane::HalfH => &self.half_h,
            Plane::HalfV => &self.half_v,
            Plane::Centre => &self.centre,
        };
        let start = (y + dy + MARGIN) as usize * self.stride + (x + dx + MARGIN) as usize;
        &data[start..start + w]
    }
}

/// The one or two half-sample reads that clause 8.4.2.2.1 averages for each
/// quarter position, in the order [`recon::qpel`] lists them.
fn reads(x_frac: i32, y_frac: i32) -> (Read, Option<Read>) {
    use Plane::{Centre, Full, HalfH, HalfV};
    match (x_frac, y_frac) {
        (0, 0) => ((Full, 0, 0), None),
        (1, 0) => ((Full, 0, 0), Some((HalfH, 0, 0))),
        (2, 0) => ((HalfH, 0, 0), None),
        (3, 0) => ((HalfH, 0, 0), Some((Full, 1, 0))),
        (0, 1) => ((Full, 0, 0), Some((HalfV, 0, 0))),
        (0, 2) => ((HalfV, 0, 0), None),
        (0, 3) => ((HalfV, 0, 0), Some((Full, 0, 1))),
        (2, 2) => ((Centre, 0, 0), None),
        (2, 1) => ((HalfH, 0, 0), Some((Centre, 0, 0))),
        (1, 2) => ((HalfV, 0, 0), Some((Centre, 0, 0))),
        (2, 3) => ((Centre, 0, 0), Some((HalfH, 0, 1))),
        (3, 2) => ((Centre, 0, 0), Some((HalfV, 1, 0))),
        (1, 1) => ((HalfH, 0, 0), Some((HalfV, 0, 0))),
        (3, 1) => ((HalfH, 0, 0), Some((HalfV, 1, 0))),
        (1, 3) => ((HalfV, 0, 0), Some((HalfH, 0, 1))),
        _ => ((HalfV, 1, 0), Some((HalfH, 0, 1))),
    }
}

/// The six-tap half-sample filter of clause 8.4.2.2.1, before rounding.
fn six(a: u8, b: u8, c: u8, d: u8, e: u8, f: u8) -> i32 {
    let p = |v: u8| i32::from(v);
    p(a) - 5 * p(b) + 20 * p(c) + 20 * p(d) - 5 * p(e) + p(f)
}

fn clip(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every quarter position of every sample, out to the margin, matches the
    /// per-sample filter the decoder uses.
    #[test]
    fn matches_qpel_everywhere() {
        // Wide enough for the unclamped inner loops, and too small for them.
        for (w, h) in [(20u32, 12u32), (7, 6), (3, 2)] {
            matches_qpel_at(w, h);
        }
    }

    fn matches_qpel_at(w: u32, h: u32) {
        let plane: Vec<u8> = (0..w * h)
            .map(|i| ((i * 97 + (i / w) * 31) % 256) as u8)
            .collect();
        let sub = Subpel::new(&plane, w, h);
        let mut out = [0u8; MAX_BLOCK * MAX_BLOCK];
        for mvy in -4 * MARGIN..4 * MARGIN {
            for mvx in -4 * MARGIN..4 * MARGIN {
                let mv = [mvx as i16, mvy as i16];
                for (bx, by, bw, bh) in [(0, 0, 4, 4), (4, 0, 16, 8), (2, 3, 8, 8)] {
                    if sub.block((bx, by, bw, bh), mv, &mut out).is_none() {
                        continue;
                    }
                    for row in 0..bh {
                        for col in 0..bw {
                            let want = recon::qpel(
                                &plane,
                                w as usize,
                                w as i32,
                                h as i32,
                                recon::Pel {
                                    x: (bx + col) as i32 + (mvx >> 2),
                                    y: (by + row) as i32 + (mvy >> 2),
                                    x_frac: mvx & 3,
                                    y_frac: mvy & 3,
                                },
                            );
                            let got = out[row as usize * MAX_BLOCK + col as usize];
                            assert_eq!(
                                got, want,
                                "{w}x{h} mv {mv:?} block {bx},{by} at {col},{row}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn refuses_reads_past_the_margin() {
        let plane = vec![0u8; 16 * 16];
        let sub = Subpel::new(&plane, 16, 16);
        let mut out = [0u8; MAX_BLOCK * MAX_BLOCK];
        let far = (4 * (MARGIN + 1)) as i16;
        assert!(sub.block((0, 0, 16, 16), [-far, 0], &mut out).is_none());
        assert!(sub.block((0, 0, 16, 16), [0, far], &mut out).is_none());
        assert!(sub.block((0, 0, 16, 16), [0, 0], &mut out).is_some());
    }
}
