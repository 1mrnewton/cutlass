//! Film grain synthesis, AV1 specification section 7.18.3.
//!
//! [`apply_film_grain`] runs the whole process on one shown frame: the
//! generate grain process (7.18.3.3) builds the luma and chroma grain
//! templates, the scaling lookup initialization process (7.18.3.4) builds the
//! scaling functions, and the add noise synthesis process (7.18.3.5) blends
//! 32x32 grain blocks with the frame.
//!
//! The specification describes the add noise process with whole-frame
//! `noiseStripe` and `noiseImage` arrays. This implementation keeps one noise
//! stripe (34 luma rows) per plane plus the previous stripe for the vertical
//! overlap, and blends each stripe into the frame as soon as it is built.
//! The scaling function is expanded to one entry per sample value, so the
//! per-sample work is a table lookup, a multiply, and a clamp.

use super::Pixel;
use crate::spec::tables::{GAUSSIAN_SEQUENCE, MC_IDENTITY};

/// The film_grain_params() syntax elements, after the parser has resolved
/// `update_grain` and `film_grain_params_ref_idx` (copying the stored
/// parameters of the reference frame and keeping this frame's `grain_seed`).
///
/// [`Default`] is the state after `reset_grain_params()`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FilmGrainParams {
    /// `apply_grain`: film grain is added to this frame.
    pub apply_grain: bool,
    /// `grain_seed`: starting value of the pseudo-random generator.
    pub grain_seed: u16,
    /// `update_grain`: a new parameter set was sent. Informational only; the
    /// caller loads the referenced parameters when this is false.
    pub update_grain: bool,
    /// `film_grain_params_ref_idx`. Informational only.
    pub film_grain_params_ref_idx: u8,
    /// `num_y_points` (0..=14).
    pub num_y_points: u8,
    /// `point_y_value[i]`, strictly increasing.
    pub point_y_value: [u8; 14],
    /// `point_y_scaling[i]`.
    pub point_y_scaling: [u8; 14],
    /// `chroma_scaling_from_luma`.
    pub chroma_scaling_from_luma: bool,
    /// `num_cb_points` (0..=10).
    pub num_cb_points: u8,
    /// `point_cb_value[i]`, strictly increasing.
    pub point_cb_value: [u8; 10],
    /// `point_cb_scaling[i]`.
    pub point_cb_scaling: [u8; 10],
    /// `num_cr_points` (0..=10).
    pub num_cr_points: u8,
    /// `point_cr_value[i]`, strictly increasing.
    pub point_cr_value: [u8; 10],
    /// `point_cr_scaling[i]`.
    pub point_cr_scaling: [u8; 10],
    /// `grain_scaling_minus_8` (0..=3).
    pub grain_scaling_minus_8: u8,
    /// `ar_coeff_lag` (0..=3).
    pub ar_coeff_lag: u8,
    /// `ar_coeffs_y_plus_128[i]`, `2 * lag * (lag + 1)` entries used.
    pub ar_coeffs_y_plus_128: [u8; 24],
    /// `ar_coeffs_cb_plus_128[i]`, one more entry than luma when
    /// `num_y_points > 0` (the luma correlation coefficient).
    pub ar_coeffs_cb_plus_128: [u8; 25],
    /// `ar_coeffs_cr_plus_128[i]`, as for cb.
    pub ar_coeffs_cr_plus_128: [u8; 25],
    /// `ar_coeff_shift_minus_6` (0..=3).
    pub ar_coeff_shift_minus_6: u8,
    /// `grain_scale_shift` (0..=3).
    pub grain_scale_shift: u8,
    /// `cb_mult`.
    pub cb_mult: u8,
    /// `cb_luma_mult`.
    pub cb_luma_mult: u8,
    /// `cb_offset` (9 bits).
    pub cb_offset: u16,
    /// `cr_mult`.
    pub cr_mult: u8,
    /// `cr_luma_mult`.
    pub cr_luma_mult: u8,
    /// `cr_offset` (9 bits).
    pub cr_offset: u16,
    /// `overlap_flag`: blend adjacent grain blocks.
    pub overlap_flag: bool,
    /// `clip_to_restricted_range`: clip to studio range after adding grain.
    pub clip_to_restricted_range: bool,
}

const LUMA_GW: usize = 82;
const LUMA_GH: usize = 73;
/// Entries in an expanded scaling table: one per 12-bit sample value.
const LUT_SIZE: usize = 4096;

/// The random number process (7.18.3.2), state `RandomRegister`.
struct Rng(u16);

impl Rng {
    #[inline(always)]
    fn bits(&mut self, bits: u32) -> usize {
        let r = self.0;
        let bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
        let r = (r >> 1) | (bit << 15);
        self.0 = r;
        usize::from((r >> (16 - bits)) & ((1 << bits) - 1))
    }
}

/// Grain templates from the generate grain process (7.18.3.3).
struct Grain {
    luma: Vec<i16>,
    cb: Vec<i16>,
    cr: Vec<i16>,
    /// chromaW, the row stride of `cb` and `cr`.
    cw: usize,
}

/// Fills `out` with white noise from the Gaussian sequence, or zeros.
fn white_noise(out: &mut [i16], seed: Option<u16>, shift: u32) {
    match seed {
        Some(seed) => {
            let mut rng = Rng(seed);
            let rnd = (1i32 << shift) >> 1;
            for g in out.iter_mut() {
                let v = i32::from(GAUSSIAN_SEQUENCE[rng.bits(11)]);
                *g = ((v + rnd) >> shift) as i16;
            }
        }
        None => out.fill(0),
    }
}

/// Auto-regressive taps preceding the current sample in raster order, as
/// (distance back in the template, coefficient).
fn ar_taps(coeffs: &[u8], lag: usize, stride: usize) -> Vec<(usize, i32)> {
    let mut taps = Vec::with_capacity(24);
    let mut pos = 0;
    for dy in 0..=lag {
        let up = lag - dy;
        for dx in 0..=2 * lag {
            if up == 0 && dx == lag {
                return taps;
            }
            let c = i32::from(coeffs[pos]) - 128;
            pos += 1;
            if c != 0 {
                taps.push((up * stride + lag - dx, c));
            }
        }
    }
    taps
}

fn generate_grain(p: &FilmGrainParams, bd: u32, sub_x: bool, sub_y: bool, mono: bool) -> Grain {
    let gmax = (128i32 << (bd - 8)) - 1;
    let gmin = -(128i32 << (bd - 8));
    let lag = usize::from(p.ar_coeff_lag.min(3));
    let shift = 12 - bd + u32::from(p.grain_scale_shift);
    let ar_shift = u32::from(p.ar_coeff_shift_minus_6) + 6;
    let ar_rnd = 1i32 << (ar_shift - 1);

    let mut luma = vec![0i16; LUMA_GW * LUMA_GH];
    let has_y = p.num_y_points > 0;
    white_noise(&mut luma, has_y.then_some(p.grain_seed), shift);
    let taps = ar_taps(&p.ar_coeffs_y_plus_128, lag, LUMA_GW);
    if has_y && !taps.is_empty() {
        for y in 3..LUMA_GH {
            for x in 3..LUMA_GW - 3 {
                let idx = y * LUMA_GW + x;
                let mut s = 0i32;
                for &(back, c) in &taps {
                    s += i32::from(luma[idx - back]) * c;
                }
                let v = i32::from(luma[idx]) + ((s + ar_rnd) >> ar_shift);
                luma[idx] = v.clamp(gmin, gmax) as i16;
            }
        }
    }

    if mono {
        return Grain {
            luma,
            cb: Vec::new(),
            cr: Vec::new(),
            cw: 0,
        };
    }
    let cw = if sub_x { 44 } else { 82 };
    let ch = if sub_y { 38 } else { 73 };
    let (sx, sy) = (usize::from(sub_x), usize::from(sub_y));
    let num_pos_luma = 2 * lag * (lag + 1);
    let mut planes = [vec![0i16; cw * ch], vec![0i16; cw * ch]];
    let setups = [
        (p.num_cb_points, 0xb524u16, &p.ar_coeffs_cb_plus_128),
        (p.num_cr_points, 0x49d8u16, &p.ar_coeffs_cr_plus_128),
    ];
    for (grain, &(num, xor, coeffs)) in planes.iter_mut().zip(&setups) {
        let used = num > 0 || p.chroma_scaling_from_luma;
        white_noise(grain, used.then_some(p.grain_seed ^ xor), shift);
        if !used {
            // Never read (the note at the end of 7.18.3.3).
            continue;
        }
        let taps = ar_taps(&coeffs[..], lag, cw);
        let luma_c = i32::from(coeffs[num_pos_luma]) - 128;
        let luma_c = if has_y { luma_c } else { 0 };
        if taps.is_empty() && luma_c == 0 {
            continue;
        }
        let lshift = sx + sy;
        let lrnd = (1i32 << lshift) >> 1;
        for y in 3..ch {
            for x in 3..cw - 3 {
                let idx = y * cw + x;
                let mut s = 0i32;
                for &(back, c) in &taps {
                    s += i32::from(grain[idx - back]) * c;
                }
                if luma_c != 0 {
                    let lx = ((x - 3) << sx) + 3;
                    let ly = ((y - 3) << sy) + 3;
                    let mut l = 0i32;
                    for i in 0..=sy {
                        for j in 0..=sx {
                            l += i32::from(luma[(ly + i) * LUMA_GW + lx + j]);
                        }
                    }
                    s += ((l + lrnd) >> lshift) * luma_c;
                }
                let v = i32::from(grain[idx]) + ((s + ar_rnd) >> ar_shift);
                grain[idx] = v.clamp(gmin, gmax) as i16;
            }
        }
    }
    let [cb, cr] = planes;
    Grain { luma, cb, cr, cw }
}

/// The scaling lookup initialization process (7.18.3.4) for one plane,
/// expanded through `scale_lut` to one entry per sample value at `bd`.
fn scaling_lut(values: &[u8], scalings: &[u8], bd: u32) -> Box<[i16; LUT_SIZE]> {
    let mut lut8 = [0i32; 256];
    let n = values.len().min(scalings.len());
    if n > 0 {
        let x0 = usize::from(values[0]);
        let y0 = i32::from(scalings[0]);
        lut8[..x0].fill(y0);
        for i in 0..n - 1 {
            let xa = i32::from(values[i]);
            let ya = i32::from(scalings[i]);
            let dx = i32::from(values[i + 1]) - xa;
            let dy = i32::from(scalings[i + 1]) - ya;
            if dx <= 0 {
                // Non-conformant (values must increase); skip the segment.
                continue;
            }
            let delta = dy * ((65536 + (dx >> 1)) / dx);
            for x in 0..dx {
                lut8[(xa + x) as usize] = ya + ((x * delta + 32768) >> 16);
            }
        }
        let xl = usize::from(values[n - 1]);
        lut8[xl..].fill(i32::from(scalings[n - 1]));
    }
    let mut lut = Box::new([0i16; LUT_SIZE]);
    let shift = bd - 8;
    let rnd = (1i32 << shift) >> 1;
    for (index, e) in lut.iter_mut().enumerate().take(1 << bd) {
        let x = index >> shift;
        let rem = (index - (x << shift)) as i32;
        let v = if shift == 0 || x == 255 {
            lut8[x]
        } else {
            let start = lut8[x];
            let end = lut8[x + 1];
            start + (((end - start) * rem + rnd) >> shift)
        };
        *e = v as i16;
    }
    lut
}

#[inline(always)]
fn blend(old: i16, new: i16, wo: i32, wn: i32, gmin: i32, gmax: i32) -> i16 {
    ((i32::from(old) * wo + i32::from(new) * wn + 16) >> 5).clamp(gmin, gmax) as i16
}

/// Builds one plane's noiseStripe (horizontal block placement and overlap).
fn build_stripe(
    grain: &[i16],
    gstride: usize,
    offsets: &[u8],
    sub_x: bool,
    sub_y: bool,
    overlap: bool,
    out: &mut [i16],
    ostride: usize,
    gmin: i32,
    gmax: i32,
) {
    let bh = 34 >> usize::from(sub_y);
    let bw = 34 >> usize::from(sub_x);
    let step = 32 >> usize::from(sub_x);
    for (bx, &rand) in offsets.iter().enumerate() {
        let (ox, oy) = (usize::from(rand >> 4), usize::from(rand & 15));
        let px = if sub_x { 6 + ox } else { 9 + ox * 2 };
        let py = if sub_y { 6 + oy } else { 9 + oy * 2 };
        let bx0 = bx * step;
        let blend_left = overlap && bx > 0;
        for i in 0..bh {
            let g = &grain[(py + i) * gstride + px..][..bw];
            let d = &mut out[i * ostride + bx0..][..bw];
            if !blend_left {
                d.copy_from_slice(g);
            } else if sub_x {
                d[0] = blend(d[0], g[0], 23, 22, gmin, gmax);
                d[1..].copy_from_slice(&g[1..]);
            } else {
                d[0] = blend(d[0], g[0], 27, 17, gmin, gmax);
                d[1] = blend(d[1], g[1], 17, 27, gmin, gmax);
                d[2..].copy_from_slice(&g[2..]);
            }
        }
    }
}

/// Vertical overlap of the first rows of `cur` with the last rows of `prev`
/// (the noiseImage construction), over the first `width` columns.
fn blend_stripes(
    cur: &mut [i16],
    prev: &[i16],
    stride: usize,
    width: usize,
    sub_y: bool,
    gmin: i32,
    gmax: i32,
) {
    let rows: &[(usize, usize, i32, i32)] = if sub_y {
        &[(0, 16, 23, 22)]
    } else {
        &[(0, 32, 27, 17), (1, 33, 17, 27)]
    };
    for &(r, pr, wo, wn) in rows {
        let c = &mut cur[r * stride..][..width];
        let o = &prev[pr * stride..][..width];
        for (c, &o) in c.iter_mut().zip(o) {
            *c = blend(o, *c, wo, wn, gmin, gmax);
        }
    }
}

#[inline(always)]
fn lut_at(lut: &[i16; LUT_SIZE], v: i32) -> i32 {
    i32::from(lut[(v as usize) & (LUT_SIZE - 1)])
}

fn apply_luma_row<P: Pixel>(
    src: &[P],
    dst: &mut [P],
    noise: &[i16],
    lut: &[i16; LUT_SIZE],
    shift: u32,
    lo: i32,
    hi: i32,
) {
    let rnd = 1i32 << (shift - 1);
    for ((d, &s), &n) in dst.iter_mut().zip(src).zip(noise) {
        let o = s.to_i32();
        let v = o + ((lut_at(lut, o) * i32::from(n) + rnd) >> shift);
        *d = P::from_i32(v.clamp(lo, hi));
    }
}

/// Per-chroma-plane constants of the add noise process.
struct ChromaCfg<'a> {
    from_luma: bool,
    mult: i32,
    luma_mult: i32,
    offset: i32,
    lut: &'a [i16; LUT_SIZE],
}

fn apply_chroma_row<P: Pixel>(
    src: &[P],
    dst: &mut [P],
    noise: &[i16],
    avg: &[i32],
    cfg: &ChromaCfg<'_>,
    shift: u32,
    lo: i32,
    hi: i32,
    px_max: i32,
) {
    let rnd = 1i32 << (shift - 1);
    let it = dst.iter_mut().zip(src).zip(noise).zip(avg);
    if cfg.from_luma {
        for (((d, &s), &n), &a) in it {
            let o = s.to_i32();
            let v = o + ((lut_at(cfg.lut, a) * i32::from(n) + rnd) >> shift);
            *d = P::from_i32(v.clamp(lo, hi));
        }
    } else {
        for (((d, &s), &n), &a) in it {
            let o = s.to_i32();
            let combined = a * cfg.luma_mult + o * cfg.mult;
            let merged = ((combined >> 6) + cfg.offset).clamp(0, px_max);
            let v = o + ((lut_at(cfg.lut, merged) * i32::from(n) + rnd) >> shift);
            *d = P::from_i32(v.clamp(lo, hi));
        }
    }
}

/// `averageLuma` for one chroma row.
fn luma_average_row<P: Pixel>(luma: &[P], avg: &mut [i32], sub_x: bool) {
    if sub_x {
        for (a, l) in avg.iter_mut().zip(luma.chunks_exact(2)) {
            *a = (l[0].to_i32() + l[1].to_i32() + 1) >> 1;
        }
        if luma.len() % 2 == 1
            && let (Some(a), Some(l)) = (avg.last_mut(), luma.last())
        {
            // Odd width: lumaNextX clamps to the last column.
            *a = l.to_i32();
        }
    } else {
        for (a, l) in avg.iter_mut().zip(luma) {
            *a = l.to_i32();
        }
    }
}

fn copy_plane<P: Pixel>(src: &[P], ss: usize, dst: &mut [P], ds: usize, w: usize, h: usize) {
    for y in 0..h {
        dst[y * ds..][..w].copy_from_slice(&src[y * ss..][..w]);
    }
}

/// Applies the film grain synthesis process (7.18.3) to a shown frame.
///
/// Reads `src_*` and writes the `w` x `h` luma area and the matching chroma
/// area of `dst_*`; planes that get no grain are copied. `w` and `h` are
/// the luma dimensions of the shown frame (UpscaledWidth and FrameHeight);
/// chroma planes are `(w + sub_x) >> sub_x` by `(h + sub_y) >> sub_y`. For
/// monochrome pass `mono = true`; the chroma slices are then not touched and
/// may be empty. `matrix_coefficients` selects the MC_IDENTITY clipping rule.
/// Source samples must be inside `0..1 << bd`. When `apply_grain` is false the
/// frame is copied unchanged.
pub fn apply_film_grain<P: Pixel>(
    params: &FilmGrainParams,
    src_y: &[P],
    src_u: &[P],
    src_v: &[P],
    src_stride_y: usize,
    src_stride_uv: usize,
    dst_y: &mut [P],
    dst_u: &mut [P],
    dst_v: &mut [P],
    dst_stride_y: usize,
    dst_stride_uv: usize,
    w: usize,
    h: usize,
    sub_x: bool,
    sub_y: bool,
    mono: bool,
    matrix_coefficients: u8,
    bd: u32,
) {
    if w == 0 || h == 0 {
        return;
    }
    let (sx, sy) = (usize::from(sub_x), usize::from(sub_y));
    let (cw, ch) = ((w + sx) >> sx, (h + sy) >> sy);
    let p = params;
    let apply_y = p.apply_grain && p.num_y_points > 0;
    let apply_u = p.apply_grain && !mono && (p.num_cb_points > 0 || p.chroma_scaling_from_luma);
    let apply_v = p.apply_grain && !mono && (p.num_cr_points > 0 || p.chroma_scaling_from_luma);
    if !apply_y {
        copy_plane(src_y, src_stride_y, dst_y, dst_stride_y, w, h);
    }
    if !mono && !apply_u {
        copy_plane(src_u, src_stride_uv, dst_u, dst_stride_uv, cw, ch);
    }
    if !mono && !apply_v {
        copy_plane(src_v, src_stride_uv, dst_v, dst_stride_uv, cw, ch);
    }
    if !(apply_y || apply_u || apply_v) {
        return;
    }

    let grain = generate_grain(p, bd, sub_x, sub_y, mono);
    let gmax = (128i32 << (bd - 8)) - 1;
    let gmin = -(128i32 << (bd - 8));
    let lut_of = |num: u8, values: &[u8], scalings: &[u8]| {
        let n = usize::from(num).min(values.len());
        scaling_lut(&values[..n], &scalings[..n], bd)
    };
    let lut_y = lut_of(p.num_y_points, &p.point_y_value, &p.point_y_scaling);
    let (lut_u, lut_v) = if p.chroma_scaling_from_luma {
        (lut_y.clone(), lut_y.clone())
    } else {
        (
            lut_of(p.num_cb_points, &p.point_cb_value, &p.point_cb_scaling),
            lut_of(p.num_cr_points, &p.point_cr_value, &p.point_cr_scaling),
        )
    };

    let hbd = bd - 8;
    let (lo, hi_y, hi_c) = if p.clip_to_restricted_range {
        let max_luma = 235 << hbd;
        let max_chroma = if i32::from(matrix_coefficients) == MC_IDENTITY {
            max_luma
        } else {
            240 << hbd
        };
        (16 << hbd, max_luma, max_chroma)
    } else {
        (0, (256 << hbd) - 1, (256 << hbd) - 1)
    };
    let px_max = (1i32 << bd) - 1;
    let shift = u32::from(p.grain_scaling_minus_8) + 8;
    let cfg_u = ChromaCfg {
        from_luma: p.chroma_scaling_from_luma,
        mult: i32::from(p.cb_mult) - 128,
        luma_mult: i32::from(p.cb_luma_mult) - 128,
        offset: (i32::from(p.cb_offset) - 256) * (1 << hbd),
        lut: &lut_u,
    };
    let cfg_v = ChromaCfg {
        from_luma: p.chroma_scaling_from_luma,
        mult: i32::from(p.cr_mult) - 128,
        luma_mult: i32::from(p.cr_luma_mult) - 128,
        offset: (i32::from(p.cr_offset) - 256) * (1 << hbd),
        lut: &lut_v,
    };

    let nbx = (w.div_ceil(2)).div_ceil(16);
    let nby = (h.div_ceil(2)).div_ceil(16);
    let ls = nbx * 32 + 2;
    let cs = if sub_x { nbx * 16 + 1 } else { nbx * 32 + 2 };
    let cbh = 34 >> sy;
    let crows = 32 >> sy;
    let mut offsets = vec![0u8; nbx];
    let alloc = |on: bool, n: usize| if on { vec![0i16; n] } else { Vec::new() };
    let (mut cur_y, mut prev_y) = (alloc(apply_y, 34 * ls), alloc(apply_y, 34 * ls));
    let (mut cur_u, mut prev_u) = (alloc(apply_u, cbh * cs), alloc(apply_u, cbh * cs));
    let (mut cur_v, mut prev_v) = (alloc(apply_v, cbh * cs), alloc(apply_v, cbh * cs));
    let mut avg = vec![0i32; if apply_u || apply_v { cw } else { 0 }];
    let overlap = p.overlap_flag;

    for s in 0..nby {
        let mut rng =
            Rng(p.grain_seed ^ ((((s * 37 + 178) & 255) << 8) | ((s * 173 + 105) & 255)) as u16);
        for o in offsets.iter_mut() {
            *o = rng.bits(8) as u8;
        }
        if apply_y {
            build_stripe(
                &grain.luma,
                LUMA_GW,
                &offsets,
                false,
                false,
                overlap,
                &mut cur_y,
                ls,
                gmin,
                gmax,
            );
            if overlap && s > 0 {
                blend_stripes(&mut cur_y, &prev_y, ls, w, false, gmin, gmax);
            }
            let y0 = s * 32;
            for y in y0..(y0 + 32).min(h) {
                apply_luma_row(
                    &src_y[y * src_stride_y..][..w],
                    &mut dst_y[y * dst_stride_y..][..w],
                    &cur_y[(y - y0) * ls..][..w],
                    &lut_y,
                    shift,
                    lo,
                    hi_y,
                );
            }
            core::mem::swap(&mut cur_y, &mut prev_y);
        }
        if !(apply_u || apply_v) {
            continue;
        }
        for (on, grain_c, cur, prev) in [
            (apply_u, &grain.cb, &mut cur_u, &prev_u),
            (apply_v, &grain.cr, &mut cur_v, &prev_v),
        ] {
            if on {
                build_stripe(
                    grain_c, grain.cw, &offsets, sub_x, sub_y, overlap, cur, cs, gmin, gmax,
                );
                if overlap && s > 0 {
                    blend_stripes(cur, prev, cs, cw, sub_y, gmin, gmax);
                }
            }
        }
        let y0 = s * crows;
        for y in y0..(y0 + crows).min(ch) {
            let ly = y << sy;
            luma_average_row(&src_y[ly * src_stride_y..][..w], &mut avg, sub_x);
            let r = (y - y0) * cs;
            if apply_u {
                apply_chroma_row(
                    &src_u[y * src_stride_uv..][..cw],
                    &mut dst_u[y * dst_stride_uv..][..cw],
                    &cur_u[r..][..cw],
                    &avg,
                    &cfg_u,
                    shift,
                    lo,
                    hi_c,
                    px_max,
                );
            }
            if apply_v {
                apply_chroma_row(
                    &src_v[y * src_stride_uv..][..cw],
                    &mut dst_v[y * dst_stride_uv..][..cw],
                    &cur_v[r..][..cw],
                    &avg,
                    &cfg_v,
                    shift,
                    lo,
                    hi_c,
                    px_max,
                );
            }
        }
        core::mem::swap(&mut cur_u, &mut prev_u);
        core::mem::swap(&mut cur_v, &mut prev_v);
    }
}

#[cfg(test)]
mod tests {
    use super::{FilmGrainParams, Pixel, apply_film_grain};
    use crate::spec::tables::{GAUSSIAN_SEQUENCE, MC_IDENTITY};

    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn r2(x: i64, n: i64) -> i64 {
        if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
    }

    /// Literal transliteration of 7.18.3 on whole-frame arrays.
    struct Spec<'a> {
        p: &'a FilmGrainParams,
        bd: i64,
        rr: i64,
        gmin: i64,
        gmax: i64,
        luma: Vec<Vec<i64>>,
        cb: Vec<Vec<i64>>,
        cr: Vec<Vec<i64>>,
        lut: [[i64; 256]; 3],
    }

    impl Spec<'_> {
        fn rand(&mut self, bits: i64) -> i64 {
            let r = self.rr;
            let bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
            let r = (r >> 1) | (bit << 15);
            self.rr = r;
            (r >> (16 - bits)) & ((1 << bits) - 1)
        }

        fn generate(&mut self, sx: usize, sy: usize, mono: bool) {
            let p = self.p;
            let shift = 12 - self.bd + p.grain_scale_shift as i64;
            self.luma = vec![vec![0; 82]; 73];
            for y in 0..73 {
                for x in 0..82 {
                    let g = if p.num_y_points > 0 {
                        GAUSSIAN_SEQUENCE[self.rand(11) as usize] as i64
                    } else {
                        0
                    };
                    self.luma[y][x] = r2(g, shift);
                }
            }
            let ars = p.ar_coeff_shift_minus_6 as i64 + 6;
            let lag = p.ar_coeff_lag as i64;
            for y in 3..73i64 {
                for x in 3..(82 - 3) as i64 {
                    let mut s = 0;
                    let mut pos = 0;
                    'outer: for dr in -lag..=0 {
                        for dc in -lag..=lag {
                            if dr == 0 && dc == 0 {
                                break 'outer;
                            }
                            let c = p.ar_coeffs_y_plus_128[pos] as i64 - 128;
                            s += self.luma[(y + dr) as usize][(x + dc) as usize] * c;
                            pos += 1;
                        }
                    }
                    let v = self.luma[y as usize][x as usize] + r2(s, ars);
                    self.luma[y as usize][x as usize] = v.clamp(self.gmin, self.gmax);
                }
            }
            if mono {
                return;
            }
            let cw = if sx == 1 { 44 } else { 82 };
            let ch = if sy == 1 { 38 } else { 73 };
            self.rr = (p.grain_seed ^ 0xb524) as i64;
            self.cb = vec![vec![0; cw]; ch];
            for y in 0..ch {
                for x in 0..cw {
                    let g = if p.num_cb_points > 0 || p.chroma_scaling_from_luma {
                        GAUSSIAN_SEQUENCE[self.rand(11) as usize] as i64
                    } else {
                        0
                    };
                    self.cb[y][x] = r2(g, shift);
                }
            }
            self.rr = (p.grain_seed ^ 0x49d8) as i64;
            self.cr = vec![vec![0; cw]; ch];
            for y in 0..ch {
                for x in 0..cw {
                    let g = if p.num_cr_points > 0 || p.chroma_scaling_from_luma {
                        GAUSSIAN_SEQUENCE[self.rand(11) as usize] as i64
                    } else {
                        0
                    };
                    self.cr[y][x] = r2(g, shift);
                }
            }
            for y in 3..ch as i64 {
                for x in 3..(cw - 3) as i64 {
                    let (mut s0, mut s1) = (0, 0);
                    let mut pos = 0;
                    'outer: for dr in -lag..=0 {
                        for dc in -lag..=lag {
                            let c0 = p.ar_coeffs_cb_plus_128[pos] as i64 - 128;
                            let c1 = p.ar_coeffs_cr_plus_128[pos] as i64 - 128;
                            if dr == 0 && dc == 0 {
                                if p.num_y_points > 0 {
                                    let mut luma = 0;
                                    let lx = (((x - 3) as usize) << sx) + 3;
                                    let ly = (((y - 3) as usize) << sy) + 3;
                                    for i in 0..=sy {
                                        for j in 0..=sx {
                                            luma += self.luma[ly + i][lx + j];
                                        }
                                    }
                                    luma = r2(luma, (sx + sy) as i64);
                                    s0 += luma * c0;
                                    s1 += luma * c1;
                                }
                                break 'outer;
                            }
                            s0 += self.cb[(y + dr) as usize][(x + dc) as usize] * c0;
                            s1 += self.cr[(y + dr) as usize][(x + dc) as usize] * c1;
                            pos += 1;
                        }
                    }
                    let (yu, xu) = (y as usize, x as usize);
                    self.cb[yu][xu] = (self.cb[yu][xu] + r2(s0, ars)).clamp(self.gmin, self.gmax);
                    self.cr[yu][xu] = (self.cr[yu][xu] + r2(s1, ars)).clamp(self.gmin, self.gmax);
                }
            }
        }

        fn point(&self, plane: usize, i: usize) -> (i64, i64) {
            let p = self.p;
            if plane == 0 || p.chroma_scaling_from_luma {
                (p.point_y_value[i] as i64, p.point_y_scaling[i] as i64)
            } else if plane == 1 {
                (p.point_cb_value[i] as i64, p.point_cb_scaling[i] as i64)
            } else {
                (p.point_cr_value[i] as i64, p.point_cr_scaling[i] as i64)
            }
        }

        fn init_lut(&mut self, num_planes: usize) {
            let p = self.p;
            for plane in 0..num_planes {
                let n = if plane == 0 || p.chroma_scaling_from_luma {
                    p.num_y_points
                } else if plane == 1 {
                    p.num_cb_points
                } else {
                    p.num_cr_points
                } as usize;
                if n == 0 {
                    self.lut[plane] = [0; 256];
                    continue;
                }
                for x in 0..self.point(plane, 0).0 {
                    self.lut[plane][x as usize] = self.point(plane, 0).1;
                }
                for i in 0..n - 1 {
                    let (xa, ya) = self.point(plane, i);
                    let (xb, yb) = self.point(plane, i + 1);
                    let (dy, dx) = (yb - ya, xb - xa);
                    let delta = dy * ((65536 + (dx >> 1)) / dx);
                    for x in 0..dx {
                        self.lut[plane][(xa + x) as usize] = ya + ((x * delta + 32768) >> 16);
                    }
                }
                for x in self.point(plane, n - 1).0..256 {
                    self.lut[plane][x as usize] = self.point(plane, n - 1).1;
                }
            }
        }

        fn scale_lut(&self, plane: usize, index: i64) -> i64 {
            let shift = self.bd - 8;
            let x = index >> shift;
            let rem = index - (x << shift);
            if self.bd == 8 || x == 255 {
                self.lut[plane][x as usize]
            } else {
                let start = self.lut[plane][x as usize];
                let end = self.lut[plane][x as usize + 1];
                start + r2((end - start) * rem, shift)
            }
        }
    }

    /// Runs 7.18.3 on `out[plane][y][x]` in place.
    fn spec_film_grain(
        p: &FilmGrainParams,
        out: &mut [Vec<Vec<i64>>; 3],
        w: usize,
        h: usize,
        sx: usize,
        sy: usize,
        mono: bool,
        mc: u8,
        bd: i64,
    ) {
        let num_planes = if mono { 1 } else { 3 };
        let gc = 128 << (bd - 8);
        let mut sp = Spec {
            p,
            bd,
            rr: p.grain_seed as i64,
            gmin: -gc,
            gmax: (256 << (bd - 8)) - 1 - gc,
            luma: Vec::new(),
            cb: Vec::new(),
            cr: Vec::new(),
            lut: [[0; 256]; 3],
        };
        sp.generate(sx, sy, mono);
        sp.init_lut(num_planes);
        let (gmin, gmax) = (sp.gmin, sp.gmax);
        let ov = p.overlap_flag;
        let sw = w + 100;
        // stripes[lumaNum][plane][i][x]
        let mut stripes: Vec<[Vec<Vec<i64>>; 3]> = Vec::new();
        let mut luma_num = 0i64;
        let mut y = 0;
        while y < h.div_ceil(2) {
            sp.rr = p.grain_seed as i64;
            sp.rr ^= ((luma_num * 37 + 178) & 255) << 8;
            sp.rr ^= (luma_num * 173 + 105) & 255;
            let mut st: [Vec<Vec<i64>>; 3] = [
                vec![vec![0; sw]; 34],
                vec![vec![0; sw]; 34],
                vec![vec![0; sw]; 34],
            ];
            let mut x = 0;
            while x < w.div_ceil(2) {
                let rand = sp.rand(8);
                let ox = (rand >> 4) as usize;
                let oy = (rand & 15) as usize;
                for plane in 0..num_planes {
                    let psx = if plane > 0 { sx } else { 0 };
                    let psy = if plane > 0 { sy } else { 0 };
                    let pox = if psx == 1 { 6 + ox } else { 9 + ox * 2 };
                    let poy = if psy == 1 { 6 + oy } else { 9 + oy * 2 };
                    for i in 0..34 >> psy {
                        for j in 0..34 >> psx {
                            let mut g = match plane {
                                0 => sp.luma[poy + i][pox + j],
                                1 => sp.cb[poy + i][pox + j],
                                _ => sp.cr[poy + i][pox + j],
                            };
                            if psx == 0 {
                                if j < 2 && ov && x > 0 {
                                    let old = st[plane][i][x * 2 + j];
                                    g = if j == 0 {
                                        old * 27 + g * 17
                                    } else {
                                        old * 17 + g * 27
                                    };
                                    g = r2(g, 5).clamp(gmin, gmax);
                                }
                                st[plane][i][x * 2 + j] = g;
                            } else {
                                if j == 0 && ov && x > 0 {
                                    let old = st[plane][i][x + j];
                                    g = old * 23 + g * 22;
                                    g = r2(g, 5).clamp(gmin, gmax);
                                }
                                st[plane][i][x + j] = g;
                            }
                        }
                    }
                }
                x += 16;
            }
            stripes.push(st);
            luma_num += 1;
            y += 16;
        }
        let mut noise: Vec<Vec<Vec<i64>>> = Vec::new();
        for plane in 0..num_planes {
            let psx = if plane > 0 { sx } else { 0 };
            let psy = if plane > 0 { sy } else { 0 };
            let ph = (h + psy) >> psy;
            let pw = (w + psx) >> psx;
            let mut img = vec![vec![0; pw]; ph];
            for y in 0..ph {
                let ln = y >> (5 - psy);
                let i = y - (ln << (5 - psy));
                for x in 0..pw {
                    let mut g = stripes[ln][plane][i][x];
                    if psy == 0 {
                        if i < 2 && ln > 0 && ov {
                            let old = stripes[ln - 1][plane][i + 32][x];
                            g = if i == 0 {
                                old * 27 + g * 17
                            } else {
                                old * 17 + g * 27
                            };
                            g = r2(g, 5).clamp(gmin, gmax);
                        }
                    } else if i < 1 && ln > 0 && ov {
                        let old = stripes[ln - 1][plane][i + 16][x];
                        g = old * 23 + g * 22;
                        g = r2(g, 5).clamp(gmin, gmax);
                    }
                    img[y][x] = g;
                }
            }
            noise.push(img);
        }
        let (min_v, max_l, max_c) = if p.clip_to_restricted_range {
            let ml = 235 << (bd - 8);
            let mcx = if mc as i32 == MC_IDENTITY {
                ml
            } else {
                240 << (bd - 8)
            };
            (16 << (bd - 8), ml, mcx)
        } else {
            (0, (256 << (bd - 8)) - 1, (256 << (bd - 8)) - 1)
        };
        let ss = p.grain_scaling_minus_8 as i64 + 8;
        let clip1 = |v: i64| v.clamp(0, (1 << bd) - 1);
        if !mono {
            for y in 0..(h + sy) >> sy {
                for x in 0..(w + sx) >> sx {
                    let lx = x << sx;
                    let ly = y << sy;
                    let lnx = (lx + 1).min(w - 1);
                    let avg = if sx == 1 {
                        r2(out[0][ly][lx] + out[0][ly][lnx], 1)
                    } else {
                        out[0][ly][lx]
                    };
                    if p.num_cb_points > 0 || p.chroma_scaling_from_luma {
                        let orig = out[1][y][x];
                        let merged = if p.chroma_scaling_from_luma {
                            avg
                        } else {
                            let comb = avg * (p.cb_luma_mult as i64 - 128)
                                + orig * (p.cb_mult as i64 - 128);
                            clip1((comb >> 6) + ((p.cb_offset as i64 - 256) << (bd - 8)))
                        };
                        let n = r2(sp.scale_lut(1, merged) * noise[1][y][x], ss);
                        out[1][y][x] = (orig + n).clamp(min_v, max_c);
                    }
                    if p.num_cr_points > 0 || p.chroma_scaling_from_luma {
                        let orig = out[2][y][x];
                        let merged = if p.chroma_scaling_from_luma {
                            avg
                        } else {
                            let comb = avg * (p.cr_luma_mult as i64 - 128)
                                + orig * (p.cr_mult as i64 - 128);
                            clip1((comb >> 6) + ((p.cr_offset as i64 - 256) << (bd - 8)))
                        };
                        let n = r2(sp.scale_lut(2, merged) * noise[2][y][x], ss);
                        out[2][y][x] = (orig + n).clamp(min_v, max_c);
                    }
                }
            }
        }
        for y in 0..h {
            for x in 0..w {
                let orig = out[0][y][x];
                let n = r2(sp.scale_lut(0, orig) * noise[0][y][x], ss);
                if p.num_y_points > 0 {
                    out[0][y][x] = (orig + n).clamp(min_v, max_l);
                }
            }
        }
    }

    fn rand_points(rng: &mut XorShift, n: usize, values: &mut [u8], scalings: &mut [u8]) {
        let mut all: Vec<u8> = (0..=255).collect();
        for i in 0..n {
            let j = i + rng.below((256 - i) as u64) as usize;
            all.swap(i, j);
        }
        let mut chosen = all[..n].to_vec();
        chosen.sort_unstable();
        for i in 0..n {
            values[i] = chosen[i];
            scalings[i] = rng.next() as u8;
        }
    }

    const FORMATS: [(bool, bool, bool); 4] = [
        (true, true, false),
        (true, false, false),
        (false, false, false),
        (false, false, true),
    ];

    fn rand_params(rng: &mut XorShift, sx: bool, sy: bool, mono: bool) -> FilmGrainParams {
        let mut p = FilmGrainParams {
            apply_grain: true,
            grain_seed: rng.next() as u16,
            update_grain: true,
            ..FilmGrainParams::default()
        };
        p.num_y_points = if rng.below(5) == 0 {
            0
        } else {
            rng.below(15) as u8
        };
        let ny = p.num_y_points as usize;
        rand_points(rng, ny, &mut p.point_y_value, &mut p.point_y_scaling);
        p.chroma_scaling_from_luma = !mono && rng.below(3) == 0;
        if !(mono || p.chroma_scaling_from_luma || (sx && sy && p.num_y_points == 0)) {
            p.num_cb_points = rng.below(11) as u8;
            p.num_cr_points = if sx && sy {
                if p.num_cb_points == 0 {
                    0
                } else {
                    1 + rng.below(10) as u8
                }
            } else {
                rng.below(11) as u8
            };
            let (ncb, ncr) = (p.num_cb_points as usize, p.num_cr_points as usize);
            rand_points(rng, ncb, &mut p.point_cb_value, &mut p.point_cb_scaling);
            rand_points(rng, ncr, &mut p.point_cr_value, &mut p.point_cr_scaling);
        }
        p.grain_scaling_minus_8 = rng.below(4) as u8;
        p.ar_coeff_lag = rng.below(4) as u8;
        let wild = rng.below(3) == 0;
        let coeff = |rng: &mut XorShift| {
            if wild {
                rng.next() as u8
            } else {
                (108 + rng.below(41)) as u8
            }
        };
        for c in p.ar_coeffs_y_plus_128.iter_mut() {
            *c = coeff(rng);
        }
        for c in p.ar_coeffs_cb_plus_128.iter_mut() {
            *c = coeff(rng);
        }
        for c in p.ar_coeffs_cr_plus_128.iter_mut() {
            *c = coeff(rng);
        }
        p.ar_coeff_shift_minus_6 = rng.below(4) as u8;
        p.grain_scale_shift = rng.below(4) as u8;
        p.cb_mult = rng.next() as u8;
        p.cb_luma_mult = rng.next() as u8;
        p.cb_offset = rng.below(512) as u16;
        p.cr_mult = rng.next() as u8;
        p.cr_luma_mult = rng.next() as u8;
        p.cr_offset = rng.below(512) as u16;
        p.overlap_flag = rng.below(2) == 1;
        p.clip_to_restricted_range = rng.below(2) == 1;
        p
    }

    fn run_case<P: Pixel>(
        rng: &mut XorShift,
        bd: u32,
        fmt: usize,
        w: usize,
        h: usize,
        p: &FilmGrainParams,
    ) {
        let (sx, sy, mono) = FORMATS[fmt];
        let mc = if rng.below(2) == 0 {
            MC_IDENTITY as u8
        } else {
            1
        };
        let (sxu, syu) = (usize::from(sx), usize::from(sy));
        let (cw, ch) = ((w + sxu) >> sxu, (h + syu) >> syu);
        let ssy = w + rng.below(9) as usize;
        let ssc = cw + rng.below(9) as usize;
        let dsy = w + rng.below(9) as usize;
        let dsc = cw + rng.below(9) as usize;
        let max = 1u64 << bd;
        let px = |n: usize, rng: &mut XorShift| -> Vec<P> {
            (0..n).map(|_| P::from_i32(rng.below(max) as i32)).collect()
        };
        let src_y = px(ssy * h, rng);
        let (src_u, src_v) = if mono {
            (Vec::new(), Vec::new())
        } else {
            (px(ssc * ch, rng), px(ssc * ch, rng))
        };
        let init_y = px(dsy * h, rng);
        let (init_u, init_v) = if mono {
            (Vec::new(), Vec::new())
        } else {
            (px(dsc * ch, rng), px(dsc * ch, rng))
        };
        let (mut dy, mut du, mut dv) = (init_y.clone(), init_u.clone(), init_v.clone());
        apply_film_grain(
            p, &src_y, &src_u, &src_v, ssy, ssc, &mut dy, &mut du, &mut dv, dsy, dsc, w, h, sx, sy,
            mono, mc, bd,
        );

        let grid = |s: &[P], st: usize, pw: usize, ph: usize| -> Vec<Vec<i64>> {
            (0..ph)
                .map(|y| (0..pw).map(|x| s[y * st + x].to_i32() as i64).collect())
                .collect()
        };
        let mut out = [
            grid(&src_y, ssy, w, h),
            if mono {
                Vec::new()
            } else {
                grid(&src_u, ssc, cw, ch)
            },
            if mono {
                Vec::new()
            } else {
                grid(&src_v, ssc, cw, ch)
            },
        ];
        if p.apply_grain {
            spec_film_grain(p, &mut out, w, h, sxu, syu, mono, mc, bd as i64);
        }
        let (mut ey, mut eu, mut ev) = (init_y, init_u, init_v);
        let put = |d: &mut [P], st: usize, g: &[Vec<i64>]| {
            for (y, row) in g.iter().enumerate() {
                for (x, &v) in row.iter().enumerate() {
                    d[y * st + x] = P::from_i32(v as i32);
                }
            }
        };
        put(&mut ey, dsy, &out[0]);
        if !mono {
            put(&mut eu, dsc, &out[1]);
            put(&mut ev, dsc, &out[2]);
        }
        assert!(dy == ey, "luma mismatch bd {bd} fmt {fmt} {w}x{h} {p:?}");
        assert!(du == eu, "cb mismatch bd {bd} fmt {fmt} {w}x{h} {p:?}");
        assert!(dv == ev, "cr mismatch bd {bd} fmt {fmt} {w}x{h} {p:?}");
    }

    fn run_bd(bd: u32, seed: u64, iters: usize) {
        let mut rng = XorShift(seed);
        for it in 0..iters {
            for (fmt, &(sx, sy, mono)) in FORMATS.iter().enumerate() {
                let mut p = rand_params(&mut rng, sx, sy, mono);
                if it % 7 == 3 {
                    p.apply_grain = false;
                }
                let (w, h) = if it % 5 == 0 {
                    (1 + rng.below(4) as usize, 1 + rng.below(4) as usize)
                } else {
                    (1 + rng.below(160) as usize, 1 + rng.below(110) as usize)
                };
                if bd == 8 {
                    run_case::<u8>(&mut rng, bd, fmt, w, h, &p);
                } else {
                    run_case::<u16>(&mut rng, bd, fmt, w, h, &p);
                }
            }
        }
    }

    #[test]
    fn matches_spec_8bit() {
        run_bd(8, 0x1234_5678_9abc_def1, 80);
    }

    #[test]
    fn matches_spec_10bit() {
        run_bd(10, 0x0fed_cba9_8765_4321, 80);
    }

    #[test]
    fn matches_spec_12bit() {
        run_bd(12, 0x5555_aaaa_3333_cccc, 80);
    }

    #[test]
    fn matches_spec_large_frames() {
        let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
        for bd in [8, 10, 12] {
            for (fmt, &(sx, sy, mono)) in FORMATS.iter().enumerate() {
                let mut p = rand_params(&mut rng, sx, sy, mono);
                p.overlap_flag = true;
                if bd == 8 {
                    run_case::<u8>(&mut rng, bd, fmt, 333, 271, &p);
                } else {
                    run_case::<u16>(&mut rng, bd, fmt, 333, 271, &p);
                }
            }
        }
    }
}
