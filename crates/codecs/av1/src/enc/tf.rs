//! Motion-compensated temporal filtering of alt-ref sources.
//!
//! A hidden alt-ref frame is predicted from by many frames, so coding a
//! denoised version of it pays off: each 16x16 luma block (and the matching
//! chroma) is averaged with the best-matching block of nearby frames,
//! weighted by how well they match.

use crate::dec::frame::Picture;
use crate::dsp::Pixel;

use super::me::{MeParams, MeScratch, motion_search};

const BLOCK: usize = 16;

/// Returns `target` filtered with `others` (each with its distance in frames).
/// `qstep` is the quantizer step in samples; stronger quantization filters more.
pub fn temporal_filter<P: Pixel>(
    target: &Picture<P>,
    others: &[(&Picture<P>, u32)],
    qstep: f64,
    bd: u32,
) -> Picture<P> {
    let mut out = target.clone();
    if others.is_empty() {
        return out;
    }
    let mut scratch = MeScratch::<P>::default();
    let params = MeParams {
        max_range: 16,
        subpel: 0,
        border: 0,
        ..MeParams::default()
    };
    let cost = |r: i32, c: i32| -> u32 { ((r.unsigned_abs() + c.unsigned_abs()) >> 3) * 4 };
    let scale = f64::from(1u32 << (2 * (bd - 8)));
    // Squared-error scale of the weights.
    let h2 = (0.6 * qstep * qstep).clamp(4.0 * scale, 900.0 * scale);
    let y = &target.planes[0];
    let rows = y.h.div_ceil(BLOCK);
    let cols = y.w.div_ceil(BLOCK);
    let np = target.planes.len();
    let (sx, sy) = if np > 1 {
        (
            usize::from(target.planes[1].w < y.w),
            usize::from(target.planes[1].h < y.h),
        )
    } else {
        (0, 0)
    };
    let mut acc = vec![0f64; BLOCK * BLOCK];
    let mut wsum = [0f64; 1];
    for br in 0..rows {
        for bc in 0..cols {
            let x0 = bc * BLOCK;
            let y0 = br * BLOCK;
            if x0 + BLOCK > y.stride || y0 + BLOCK > y.alloc_h {
                continue;
            }
            // Motion and weight of each other frame for this block.
            let mut matches: Vec<(usize, i32, i32, f64)> = Vec::with_capacity(others.len());
            for (k, (o, dist)) in others.iter().enumerate() {
                let op = &o.planes[0];
                let r = motion_search(
                    &mut scratch,
                    &y.data,
                    y.stride,
                    x0,
                    y0,
                    BLOCK,
                    BLOCK,
                    &op.data,
                    op.stride,
                    op.w,
                    op.h,
                    &[(0, 0)],
                    &params,
                    &cost,
                    bd,
                );
                let dy = r.mv.0 >> 3;
                let dx = r.mv.1 >> 3;
                let mut sse = 0f64;
                for i in 0..BLOCK {
                    let ry = clampi(y0 as i32 + i as i32 + dy, op.alloc_h);
                    for j in 0..BLOCK {
                        let rx = clampi(x0 as i32 + j as i32 + dx, op.stride);
                        let d = f64::from(
                            y.data[(y0 + i) * y.stride + x0 + j].to_i32()
                                - op.data[ry * op.stride + rx].to_i32(),
                        );
                        sse += d * d;
                    }
                }
                let mse = sse / (BLOCK * BLOCK) as f64;
                let decay = if *dist <= 1 { 1.0 } else { 0.75 };
                let w = (-mse / h2).exp() * decay;
                if w > 0.02 {
                    matches.push((k, dy, dx, w));
                }
            }
            if matches.is_empty() {
                continue;
            }
            for p in 0..np {
                let (px, py) = if p == 0 { (0, 0) } else { (sx, sy) };
                let bw = BLOCK >> px;
                let bh = BLOCK >> py;
                let xs = x0 >> px;
                let ys = y0 >> py;
                let tp = &target.planes[p];
                acc[..bw * bh].fill(0.0);
                wsum[0] = 1.0;
                for i in 0..bh {
                    for j in 0..bw {
                        acc[i * bw + j] =
                            f64::from(tp.data[(ys + i) * tp.stride + xs + j].to_i32());
                    }
                }
                for &(k, dy, dx, w) in &matches {
                    let op = &others[k].0.planes[p];
                    let (dy, dx) = (dy >> py, dx >> px);
                    for i in 0..bh {
                        let ry = clampi((ys + i) as i32 + dy, op.alloc_h);
                        for j in 0..bw {
                            let rx = clampi((xs + j) as i32 + dx, op.stride);
                            acc[i * bw + j] += w * f64::from(op.data[ry * op.stride + rx].to_i32());
                        }
                    }
                    wsum[0] += w;
                }
                let dst = &mut out.planes[p];
                let inv = 1.0 / wsum[0];
                for i in 0..bh {
                    for j in 0..bw {
                        dst.data[(ys + i) * dst.stride + xs + j] =
                            P::from_i32((acc[i * bw + j] * inv).round() as i32);
                    }
                }
            }
        }
    }
    out
}

fn clampi(v: i32, n: usize) -> usize {
    v.clamp(0, n as i32 - 1) as usize
}
