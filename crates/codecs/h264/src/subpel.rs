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
        let mut raw = vec![0i32; stride * raw_rows];
        for ry in 0..raw_rows {
            let y = ry as i32 - MARGIN - 2;
            for rx in 0..stride {
                let x = rx as i32 - MARGIN;
                raw[ry * stride + rx] = recon::tap(plane, src_stride, w, h, x, y, true);
            }
        }
        let mut full = vec![0u8; stride * rows];
        let mut half_h = vec![0u8; stride * rows];
        let mut half_v = vec![0u8; stride * rows];
        let mut centre = vec![0u8; stride * rows];
        for row in 0..rows {
            let y = row as i32 - MARGIN;
            for col in 0..stride {
                let x = col as i32 - MARGIN;
                let index = row * stride + col;
                full[index] = clip(recon::sample(plane, src_stride, w, h, x, y));
                half_h[index] = clip((raw[(row + 2) * stride + col] + 16) >> 5);
                half_v[index] =
                    clip((recon::tap(plane, src_stride, w, h, x, y, false) + 16) >> 5);
                let tap = |k: usize| raw[(row + k) * stride + col];
                let acc = tap(0) - 5 * tap(1) + 20 * tap(2) + 20 * tap(3) - 5 * tap(4) + tap(5);
                centre[index] = clip((acc + 512) >> 10);
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
        let (w, h) = (20u32, 12u32);
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
                            assert_eq!(got, want, "mv {mv:?} block {bx},{by} at {col},{row}");
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
