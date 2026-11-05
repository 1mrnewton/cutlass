//! Loop restoration (Wiener) per unit, chosen after the frame is coded.
//!
//! For each restoration unit a separable symmetric Wiener filter is fitted
//! by least squares (horizontal and vertical passes alternately) from the
//! CDEF output to the source. The whole frame is then restored once with
//! every unit's filter, and each unit keeps its filter only where the
//! error it removes pays for its coefficient bits. The syntax goes into
//! the tiles when their logged symbols are coded again
//! ([`LrWriter::write_sb`] at each superblock start).

use crate::dec::frame::{FrameState, LrUnitParams, Picture};
use crate::dec::obu::{FrameHeader, SequenceHeader};
use crate::dec::postfilter::loop_restoration;
use crate::dsp::Pixel;
use crate::dsp::lr::{LrScratch, count_units_in_frame};
use crate::spec::tables::{RESTORE_NONE, RESTORE_WIENER};

use super::msac_enc::MsacEnc;

const WMIN: [i32; 3] = [-5, -23, -17];
const WMAX: [i32; 3] = [10, 8, 46];
const WK: [u32; 3] = [1, 2, 3];
/// The reference coefficients each tile starts from.
const WIENER_REF: [i8; 3] = [3, -7, 15];

/// The restoration chosen for a frame.
#[derive(Clone, Debug, Default)]
pub struct LrPlan {
    pub frame_type: [u8; 3],
    pub unit_size: [u32; 3],
    pub units: [Vec<LrUnitParams>; 3],
    pub cols: [usize; 3],
    pub rows: [usize; 3],
}

impl LrPlan {
    /// Writes the plan into a frame header.
    pub fn apply(&self, h: &mut FrameHeader) {
        h.lr.frame_type = self.frame_type;
        h.lr.unit_size = self.unit_size;
        h.lr.uses_lr = self.frame_type.iter().any(|&t| t != RESTORE_NONE as u8);
    }
}

/// Number of bits `decode_subexp_bool` spends on `value` (all of them
/// equiprobable).
fn subexp_bits(num_syms: u32, k: u32, value: u32) -> u32 {
    let mut i = 0u32;
    let mut mk = 0u32;
    let mut bits = 0;
    loop {
        let b2 = if i != 0 { k + i - 1 } else { k };
        let a = 1u32 << b2;
        if num_syms <= mk + 3 * a {
            let n = num_syms - mk;
            if n <= 1 {
                return bits;
            }
            let w = 32 - n.leading_zeros();
            let m = (1u32 << w) - n;
            return bits + if value.wrapping_sub(mk) < m { w - 1 } else { w };
        }
        bits += 1;
        if value >= mk + a {
            i += 1;
            mk += a;
        } else {
            return bits + b2;
        }
    }
}

fn recenter(r: i32, v: i32) -> i32 {
    if v > 2 * r {
        v
    } else if v >= r {
        (v - r) * 2
    } else {
        (r - v) * 2 - 1
    }
}

/// The value `decode_signed_subexp_with_ref_bool` codes for `target`.
fn subexp_value(low: i32, high: i32, r: i32, target: i32) -> (u32, u32) {
    let mx = high - low;
    let r = r - low;
    let x = target - low;
    let v = if (r << 1) <= mx {
        recenter(r, x)
    } else {
        recenter(mx - 1 - r, mx - 1 - x)
    };
    (mx as u32, v.max(0) as u32)
}

/// Bits of a unit's Wiener coefficients against the reference.
fn wiener_bits(w: &[[i8; 3]; 2], refw: &[[i8; 3]; 2], chroma: bool) -> u32 {
    let mut bits = 0;
    for pass in 0..2 {
        for j in usize::from(chroma)..3 {
            let (n, v) = subexp_value(
                WMIN[j],
                WMAX[j] + 1,
                i32::from(refw[pass][j]),
                i32::from(w[pass][j]),
            );
            bits += subexp_bits(n, WK[j], v);
        }
    }
    bits
}

/// Solves the 3x3 system `m x = b` (Gaussian elimination, partial pivots).
fn solve3(mut m: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for c in 0..3 {
        let p = (c..3).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))?;
        if m[p][c].abs() < 1e-9 {
            return None;
        }
        m.swap(c, p);
        b.swap(c, p);
        for r in 0..3 {
            if r != c {
                let f = m[r][c] / m[c][c];
                for k in 0..3 {
                    m[r][k] -= f * m[c][k];
                }
                b[r] -= f * b[c];
            }
        }
    }
    Some([b[0] / m[0][0], b[1] / m[1][1], b[2] / m[2][2]])
}

/// Least-squares symmetric 1-D filter (offsets 3, 2, 1 from the center,
/// taps summing to one) along `dx, dy` from `x` to `y` over `w x h`
/// samples; `x` has a margin of 3 on every side (stride `xs`).
fn fit_pass(
    x: &[f64],
    xs: usize,
    y: &[f64],
    w: usize,
    h: usize,
    horz: bool,
    chroma: bool,
) -> [f64; 3] {
    let mut m = [[0f64; 3]; 3];
    let mut b = [0f64; 3];
    for i in 0..h {
        for j in 0..w {
            let at = |o: isize| -> f64 {
                let (ii, jj) = if horz {
                    (i as isize + 3, j as isize + 3 + o)
                } else {
                    (i as isize + 3 + o, j as isize + 3)
                };
                x[ii as usize * xs + jj as usize]
            };
            let c = at(0);
            let d = [
                at(-3) + at(3) - 2.0 * c,
                at(-2) + at(2) - 2.0 * c,
                at(-1) + at(1) - 2.0 * c,
            ];
            let e = y[i * w + j] - c;
            for r in 0..3 {
                for k in 0..3 {
                    m[r][k] += d[r] * d[k];
                }
                b[r] += d[r] * e;
            }
        }
    }
    if chroma {
        // The outermost tap is zero (5-tap filter).
        for k in 0..3 {
            m[0][k] = 0.0;
            m[k][0] = 0.0;
        }
        m[0][0] = 1.0;
        b[0] = 0.0;
    }
    // A little regularization keeps flat units sane.
    let n = (w * h) as f64;
    for (k, row) in m.iter_mut().enumerate() {
        row[k] += 1e-3 * n;
    }
    solve3(m, b).unwrap_or([0.0; 3])
}

/// Applies a symmetric 1-D filter `a` (offsets 3, 2, 1) to the inner part
/// of `x` (margin 3), into `out` with the same layout and margin.
fn apply_pass(x: &[f64], xs: usize, xh: usize, a: [f64; 3], horz: bool, out: &mut [f64]) {
    let c0 = 1.0 - 2.0 * (a[0] + a[1] + a[2]);
    out.copy_from_slice(x);
    let (r0, r1, c0r, c1r) = if horz {
        (0, xh, 3, xs - 3)
    } else {
        (3, xh - 3, 0, xs)
    };
    for i in r0..r1 {
        for j in c0r..c1r {
            let at = |o: isize| -> f64 {
                if horz {
                    x[i * xs + (j as isize + o) as usize]
                } else {
                    x[(i as isize + o) as usize * xs + j]
                }
            };
            out[i * xs + j] = c0 * at(0)
                + a[0] * (at(-3) + at(3))
                + a[1] * (at(-2) + at(2))
                + a[2] * (at(-1) + at(1));
        }
    }
}

/// Fits the Wiener coefficients of one unit (plane `p`, samples
/// `x0..x1, y0..y1`).
#[allow(clippy::too_many_arguments)]
fn fit_unit<P: Pixel>(
    cdef: &Picture<P>,
    src: &Picture<P>,
    p: usize,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    pw: usize,
    ph: usize,
) -> [[i8; 3]; 2] {
    let chroma = p > 0;
    let (w, h) = (x1 - x0, y1 - y0);
    let xs = w + 6;
    let xh = h + 6;
    let cp = &cdef.planes[p];
    let sp = &src.planes[p];
    let mut x = vec![0f64; xs * xh];
    for i in 0..xh {
        let yy = (y0 as isize + i as isize - 3).clamp(0, ph as isize - 1) as usize;
        for j in 0..xs {
            let xx = (x0 as isize + j as isize - 3).clamp(0, pw as isize - 1) as usize;
            x[i * xs + j] = f64::from(cp.data[yy * cp.stride + xx].to_i32());
        }
    }
    let mut y = vec![0f64; w * h];
    for i in 0..h {
        for j in 0..w {
            y[i * w + j] = f64::from(sp.data[(y0 + i) * sp.stride + x0 + j].to_i32());
        }
    }
    let mut v = [0f64; 3];
    let mut hz = [0f64; 3];
    let mut tmp = vec![0f64; xs * xh];
    for _ in 0..2 {
        // Horizontal given vertical.
        apply_pass(&x, xs, xh, v, false, &mut tmp);
        hz = fit_pass(&tmp, xs, &y, w, h, true, chroma);
        // Vertical given horizontal.
        apply_pass(&x, xs, xh, hz, true, &mut tmp);
        v = fit_pass(&tmp, xs, &y, w, h, false, chroma);
    }
    let q = |a: [f64; 3]| -> [i8; 3] {
        let mut o = [0i8; 3];
        for k in 0..3 {
            o[k] = ((a[k] * 128.0).round() as i32).clamp(WMIN[k], WMAX[k]) as i8;
        }
        if chroma {
            o[0] = 0;
        }
        o
    };
    [q(v), q(hz)]
}

/// Squared error of each unit of plane `p` between `a` and `b`.
fn unit_sse<P: Pixel>(
    a: &Picture<P>,
    b: &Picture<P>,
    p: usize,
    size: usize,
    cols: usize,
    rows: usize,
    pw: usize,
    ph: usize,
) -> Vec<u64> {
    let mut out = vec![0u64; cols * rows];
    let (pa, pb) = (&a.planes[p], &b.planes[p]);
    for y in 0..ph {
        let ur = (y / size).min(rows - 1);
        for x in 0..pw {
            let uc = (x / size).min(cols - 1);
            let d = i64::from(
                pa.data[y * pa.stride + x].to_i32() - pb.data[y * pb.stride + x].to_i32(),
            );
            out[ur * cols + uc] += (d * d) as u64;
        }
    }
    out
}

/// Chooses Wiener filters for the frame coded with header `hdr` whose CDEF
/// output is `cdef` (deblocked: `deblocked`). `lambda` converts a bit into
/// squared error. `fs` gets the plan's units (it is left with them).
#[allow(clippy::too_many_arguments)]
pub fn search<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &mut FrameState,
    cdef: &Picture<P>,
    deblocked: &Picture<P>,
    src: &Picture<P>,
    lambda: f64,
    unit_size: u32,
    threads: usize,
) -> Option<LrPlan> {
    let np = seq.num_planes();
    let mut plan = LrPlan::default();
    let mut h2 = hdr.clone();
    for p in 0..np {
        let (sx, sy) = if p > 0 {
            (
                usize::from(seq.subsampling_x),
                usize::from(seq.subsampling_y),
            )
        } else {
            (0, 0)
        };
        let size = unit_size as usize;
        let pw = (hdr.upscaled_width as usize + sx) >> sx;
        let ph = (hdr.frame_height as usize + sy) >> sy;
        let cols = count_units_in_frame(size, pw);
        let rows = count_units_in_frame(size, ph);
        plan.unit_size[p] = unit_size;
        plan.cols[p] = cols;
        plan.rows[p] = rows;
        let mut units = vec![LrUnitParams::default(); cols * rows];
        for ur in 0..rows {
            for uc in 0..cols {
                let x0 = uc * size;
                let y0 = ur * size;
                let x1 = if uc + 1 == cols { pw } else { x0 + size };
                let y1 = if ur + 1 == rows { ph } else { y0 + size };
                let u = &mut units[ur * cols + uc];
                u.kind = RESTORE_WIENER as u8;
                u.wiener = fit_unit(cdef, src, p, x0, x1, y0, y1, pw, ph);
            }
        }
        plan.units[p] = units;
        plan.frame_type[p] = RESTORE_WIENER as u8;
    }
    // Restore with every unit's filter.
    plan.apply(&mut h2);
    fs.lr = plan.units.clone();
    fs.lr_cols = plan.cols;
    fs.lr_rows = plan.rows;
    let mut out = cdef.clone();
    let mut scratch = LrScratch::new();
    loop_restoration(
        seq,
        &h2,
        fs,
        cdef,
        deblocked,
        &mut out,
        &mut scratch,
        threads,
    );
    for p in 0..np {
        let (sx, sy) = if p > 0 {
            (
                usize::from(seq.subsampling_x),
                usize::from(seq.subsampling_y),
            )
        } else {
            (0, 0)
        };
        let pw = (hdr.upscaled_width as usize + sx) >> sx;
        let ph = (hdr.frame_height as usize + sy) >> sy;
        let (size, cols, rows) = (unit_size as usize, plan.cols[p], plan.rows[p]);
        let base = unit_sse(cdef, src, p, size, cols, rows, pw, ph);
        let filt = unit_sse(&out, src, p, size, cols, rows, pw, ph);
        // Units in coding order (raster order with superblock-sized or
        // larger units), each against the last coefficients coded.
        let mut refw = [WIENER_REF; 2];
        let mut any = false;
        for (k, u) in plan.units[p].iter_mut().enumerate() {
            let bits = 1 + wiener_bits(&u.wiener, &refw, p > 0);
            let gain = base[k] as f64 - filt[k] as f64;
            if gain > lambda * f64::from(bits) {
                refw = u.wiener;
                any = true;
            } else {
                *u = LrUnitParams::default();
            }
        }
        if !any {
            plan.frame_type[p] = RESTORE_NONE as u8;
            plan.units[p]
                .iter_mut()
                .for_each(|u| *u = LrUnitParams::default());
        }
    }
    plan.frame_type
        .iter()
        .any(|&t| t != RESTORE_NONE as u8)
        .then_some(plan)
}

/// Codes a plan's unit syntax at superblock starts (`read_lr`), with the
/// contexts and references a tile starts from.
pub struct LrWriter<'a> {
    plan: &'a LrPlan,
    use_wiener: [u16; 2],
    ref_wiener: [[[i8; 3]; 2]; 3],
    sub: (u32, u32),
    sb_mi: usize,
}

impl<'a> LrWriter<'a> {
    pub fn new(plan: &'a LrPlan, use_wiener: [u16; 2], seq: &SequenceHeader) -> Self {
        Self {
            plan,
            use_wiener,
            ref_wiener: [[WIENER_REF; 2]; 3],
            sub: (u32::from(seq.subsampling_x), u32::from(seq.subsampling_y)),
            sb_mi: if seq.use_128x128_superblock { 32 } else { 16 },
        }
    }

    /// read_lr() of the superblock at mi (`r`, `c`).
    pub fn write_sb(&mut self, e: &mut MsacEnc, r: usize, c: usize) {
        let (w, h) = (self.sb_mi, self.sb_mi);
        for plane in 0..3 {
            if self.plan.frame_type[plane] != RESTORE_WIENER as u8 {
                continue;
            }
            let (sx, sy) = if plane > 0 { self.sub } else { (0, 0) };
            let unit_size = self.plan.unit_size[plane] as usize;
            let rows = self.plan.rows[plane];
            let cols = self.plan.cols[plane];
            let row_start = (r * (4 >> sy)).div_ceil(unit_size);
            let row_end = rows.min(((r + h) * (4 >> sy)).div_ceil(unit_size));
            let col_start = (c * (4 >> sx)).div_ceil(unit_size);
            let col_end = cols.min(((c + w) * (4 >> sx)).div_ceil(unit_size));
            for ur in row_start..row_end {
                for uc in col_start..col_end {
                    let u = self.plan.units[plane][ur * cols + uc];
                    let on = u.kind == RESTORE_WIENER as u8;
                    e.bool_adapt(&mut self.use_wiener, on);
                    if on {
                        for pass in 0..2 {
                            for j in usize::from(plane > 0)..3 {
                                let (n, v) = subexp_value(
                                    WMIN[j],
                                    WMAX[j] + 1,
                                    i32::from(self.ref_wiener[plane][pass][j]),
                                    i32::from(u.wiener[pass][j]),
                                );
                                write_subexp(e, n, WK[j], v);
                                self.ref_wiener[plane][pass][j] = u.wiener[pass][j];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// decode_subexp_bool written.
fn write_subexp(e: &mut MsacEnc, num_syms: u32, k: u32, value: u32) {
    let mut i = 0u32;
    let mut mk = 0u32;
    loop {
        let b2 = if i != 0 { k + i - 1 } else { k };
        let a = 1u32 << b2;
        if num_syms <= mk + 3 * a {
            e.ns(num_syms - mk, value.wrapping_sub(mk));
            return;
        }
        let more = value >= mk + a;
        e.literal(1, u32::from(more));
        if more {
            i += 1;
            mk += a;
        } else {
            e.literal(b2, value.wrapping_sub(mk));
            return;
        }
    }
}
