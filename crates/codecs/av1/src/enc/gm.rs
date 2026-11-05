//! Global motion estimation: a rotation-zoom or affine model per reference,
//! fitted robustly to block motion vectors.

use crate::dec::frame::Picture;
use crate::dec::obu::{AFFINE, IDENTITY, ROTZOOM};
use crate::dsp::Pixel;
use crate::dsp::mc::setup_shear;

use super::me::{MeParams, MeScratch, motion_search};

const BLOCK: usize = 16;
const PREC: f64 = 65536.0;

/// One block correspondence: source block center and its position in the reference.
#[derive(Clone, Copy)]
struct Pt {
    x: f64,
    y: f64,
    dx: f64,
    dy: f64,
}

/// Returns the global motion type and parameters (AV1 `gm_params` layout,
/// 1/65536 units) for predicting `src` from `refp`, or `None` when no
/// model beats per-block translation.
pub fn estimate<P: Pixel>(src: &Picture<P>, refp: &Picture<P>, bd: u32) -> Option<(u8, [i32; 6])> {
    let sp = &src.planes[0];
    let rp = &refp.planes[0];
    if sp.w < 4 * BLOCK || sp.h < 4 * BLOCK {
        return None;
    }
    let mut scratch = MeScratch::<P>::default();
    let params = MeParams {
        max_range: 24,
        subpel: 2,
        ..MeParams::default()
    };
    let cost = |r: i32, c: i32| -> u32 { (r.unsigned_abs() + c.unsigned_abs()) >> 2 };
    let mut pts = Vec::new();
    let mut y0 = BLOCK;
    while y0 + 2 * BLOCK <= sp.h {
        let mut x0 = BLOCK;
        while x0 + 2 * BLOCK <= sp.w {
            // Skip flat blocks: their motion is not reliable.
            let mut lo = i32::MAX;
            let mut hi = i32::MIN;
            for i in 0..BLOCK {
                for j in 0..BLOCK {
                    let v = sp.data[(y0 + i) * sp.stride + x0 + j].to_i32();
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            if hi - lo >= (12 << (bd - 8)) {
                let r = motion_search(
                    &mut scratch,
                    &sp.data,
                    sp.stride,
                    x0,
                    y0,
                    BLOCK,
                    BLOCK,
                    &rp.data,
                    rp.stride,
                    rp.w,
                    rp.h,
                    &[(0, 0)],
                    &params,
                    &cost,
                    bd,
                );
                let cx = x0 as f64 + BLOCK as f64 / 2.0 - 0.5;
                let cy = y0 as f64 + BLOCK as f64 / 2.0 - 0.5;
                pts.push(Pt {
                    x: cx,
                    y: cy,
                    dx: cx + f64::from(r.mv.1) / 8.0,
                    dy: cy + f64::from(r.mv.0) / 8.0,
                });
            }
            x0 += BLOCK;
        }
        y0 += BLOCK;
    }
    if pts.len() < 12 {
        return None;
    }
    let mut best: Option<(u8, [f64; 6], usize)> = None;
    for ty in [ROTZOOM, AFFINE] {
        let mut inl: Vec<Pt> = pts.clone();
        let mut model = fit(&inl, ty)?;
        for _ in 0..4 {
            let mut errs: Vec<f64> = pts.iter().map(|p| err(&model, p)).collect();
            errs.sort_by(f64::total_cmp);
            let thr = (errs[errs.len() / 2] * 2.0).max(0.5);
            inl = pts
                .iter()
                .copied()
                .filter(|p| err(&model, p) <= thr)
                .collect();
            if inl.len() < 8 {
                break;
            }
            model = fit(&inl, ty)?;
        }
        let n = pts.iter().filter(|p| err(&model, p) <= 1.0).count();
        if best.as_ref().is_none_or(|b| n > b.2 + pts.len() / 20) {
            best = Some((ty, model, n));
        }
    }
    let (ty, m, n) = best?;
    // Most blocks must follow the model, and it must not be a plain translation.
    if n * 2 < pts.len() {
        return None;
    }
    let nontrivial = (m[2] - 1.0).abs() > 0.002
        || m[3].abs() > 0.002
        || m[4].abs() > 0.002
        || (m[5] - 1.0).abs() > 0.002;
    if !nontrivial {
        return None;
    }
    let q = quantize(&m, ty)?;
    setup_shear(&q)?;
    Some((ty, q))
}

/// Distance between the model's prediction and the measured position.
fn err(m: &[f64; 6], p: &Pt) -> f64 {
    let x = m[2] * p.x + m[3] * p.y + m[0];
    let y = m[4] * p.x + m[5] * p.y + m[1];
    ((x - p.dx).powi(2) + (y - p.dy).powi(2)).sqrt()
}

/// Least-squares fit of x' = m2 x + m3 y + m0, y' = m4 x + m5 y + m1
/// (rotation-zoom ties m4 = -m3, m5 = m2).
fn fit(pts: &[Pt], ty: u8) -> Option<[f64; 6]> {
    if ty == ROTZOOM {
        // Unknowns (a, b, tx, ty): x' = a x + b y + tx, y' = -b x + a y + ty.
        let mut ata = [[0f64; 4]; 4];
        let mut atb = [0f64; 4];
        for p in pts {
            let rows = [([p.x, p.y, 1.0, 0.0], p.dx), ([p.y, -p.x, 0.0, 1.0], p.dy)];
            for (r, v) in rows {
                for i in 0..4 {
                    for j in 0..4 {
                        ata[i][j] += r[i] * r[j];
                    }
                    atb[i] += r[i] * v;
                }
            }
        }
        let s = solve4(ata, atb)?;
        Some([s[2], s[3], s[0], s[1], -s[1], s[0]])
    } else {
        // Two independent 3-parameter fits.
        let mut ata = [[0f64; 3]; 3];
        let mut bx = [0f64; 3];
        let mut by = [0f64; 3];
        for p in pts {
            let r = [p.x, p.y, 1.0];
            for i in 0..3 {
                for j in 0..3 {
                    ata[i][j] += r[i] * r[j];
                }
                bx[i] += r[i] * p.dx;
                by[i] += r[i] * p.dy;
            }
        }
        let sx = solve3(ata, bx)?;
        let sy = solve3(ata, by)?;
        Some([sx[2], sy[2], sx[0], sx[1], sy[0], sy[1]])
    }
}

fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let mut m = [[0f64; 4]; 4];
    let mut v = [0f64; 4];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = a[i][j];
        }
        v[i] = b[i];
    }
    m[3][3] = 1.0;
    let s = solve4(m, v)?;
    Some([s[0], s[1], s[2]])
}

/// Gaussian elimination with partial pivoting.
fn solve4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> Option<[f64; 4]> {
    for c in 0..4 {
        let p = (c..4).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-9 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in 0..4 {
            if r != c {
                let f = a[r][c] / a[c][c];
                for k in c..4 {
                    a[r][k] -= f * a[c][k];
                }
                b[r] -= f * b[c];
            }
        }
    }
    Some([
        b[0] / a[0][0],
        b[1] / a[1][1],
        b[2] / a[2][2],
        b[3] / a[3][3],
    ])
}

/// Rounds the model to the precision and range the bitstream codes
/// (translation 1/64 sample within +-64, matrix 1/32768 within +-1/8 of identity).
fn quantize(m: &[f64; 6], ty: u8) -> Option<[i32; 6]> {
    let mut q = [0i32; 6];
    for i in 0..6 {
        let base = if i == 2 || i == 5 { PREC } else { 0.0 };
        let (step, lim) = if i < 2 {
            (1024.0, 4096.0)
        } else {
            (2.0, 4096.0)
        };
        let v = ((m[i] * PREC - base) / step).round();
        if v.abs() > lim {
            return None;
        }
        q[i] = (v * step + base) as i32;
    }
    if ty == ROTZOOM {
        q[4] = -q[3];
        q[5] = q[2];
    }
    if ty == IDENTITY {
        return None;
    }
    Some(q)
}
