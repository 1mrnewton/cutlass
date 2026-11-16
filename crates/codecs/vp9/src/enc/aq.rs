//! Adaptive quantization by segment (libvpx's `aq-mode=1`, variance AQ):
//! each 32×32 region of the picture goes in one of five segments by how
//! busy its luma is against the frame's average, and each segment's
//! quantizer is moved from the frame's: flat regions, where coding
//! errors show, finer; busy ones, which mask them, coarser. It spends
//! bits where they are seen and gives up a little PSNR for it.

use crate::dsp::pixel::Pixel;
use crate::prob;

/// The segments used, from flattest to busiest.
pub const SEGMENTS: usize = 5;

/// Each segment's quantizer change, as a fraction of the frame's index.
const DELTA: [f64; SEGMENTS] = [-0.15, -0.07, 0.0, 0.07, 0.15];

/// A frame's segmentation: the segment of every 8×8 block, each
/// segment's quantizer delta, and the probabilities its ids are coded at.
pub struct Plan {
    pub map: Vec<u8>,
    pub deltas: [i16; 8],
    pub tree_probs: [u8; 7],
    /// Each segment's loop filter level delta, reference (0 intra, 1
    /// LAST, 2 GOLDEN, 3 ALTREF) and skip (no residual, and in an inter
    /// frame the zero vector), when it has them (a region of interest
    /// map's).
    pub lf: [i8; 8],
    pub refs: [Option<u8>; 8],
    pub skip: [bool; 8],
    /// Every block keeps to one segment (see `tile::Shared::seg_exact`).
    pub exact: bool,
    /// Each segment's blocks weigh rate against error at their own
    /// quantizer (see `tile::Shared::seg_lam`).
    pub seg_lambda: bool,
}

/// The plan for luma `y` (rows `stride` apart, `w`×`h` visible) at
/// quantizer index `qindex`, on a grid of `mi_cols`×`mi_rows` 8×8 blocks.
pub fn plan<P: Pixel>(
    y: &[P],
    stride: usize,
    (w, h): (usize, usize),
    (mi_cols, mi_rows): (usize, usize),
    qindex: u8,
) -> Plan {
    // The log variance of each 32×32 region.
    let (rc, rr) = (mi_cols.div_ceil(4), mi_rows.div_ceil(4));
    let mut energy = vec![0f64; rc * rr];
    for ry in 0..rr {
        for rx in 0..rc {
            let (x0, y0) = (rx * 32, ry * 32);
            let (x1, y1) = ((x0 + 32).min(w), (y0 + 32).min(h));
            let (mut sum, mut sq, mut n) = (0u64, 0u64, 0u64);
            for yy in y0..y1 {
                for &v in &y[yy * stride + x0..yy * stride + x1] {
                    let v = v.get() as u64;
                    sum += v;
                    sq += v * v;
                    n += 1;
                }
            }
            let var = if n == 0 {
                0.0
            } else {
                (sq as f64 - (sum as f64) * (sum as f64) / n as f64) / n as f64
            };
            energy[ry * rc + rx] = (var + 1.0).log2();
        }
    }
    let avg = energy.iter().sum::<f64>() / energy.len().max(1) as f64;
    let mut map = vec![0u8; mi_cols * mi_rows];
    let mut used = [0u32; 8];
    for r in 0..mi_rows {
        for c in 0..mi_cols {
            let e = energy[(r / 4) * rc + c / 4] - avg;
            let seg = (e.round().clamp(-2.0, 2.0) + 2.0) as u8;
            map[r * mi_cols + c] = seg;
            used[usize::from(seg)] += 1;
        }
    }
    let mut deltas = [0i16; 8];
    for (d, f) in deltas.iter_mut().zip(DELTA) {
        let q = (f64::from(qindex) * (1.0 + f)).round().clamp(1.0, 255.0);
        *d = (q - f64::from(qindex)) as i16;
    }
    Plan {
        map,
        deltas,
        tree_probs: tree_probs(&used),
        lf: [0; 8],
        refs: [None; 8],
        skip: [false; 8],
        exact: false,
        seg_lambda: false,
    }
}

impl Plan {
    /// A plan of the segment `map` (one id an 8×8 block) with these
    /// quantizer deltas.
    pub fn of_map(map: Vec<u8>, deltas: [i16; 8]) -> Self {
        let mut used = [0u32; 8];
        for &s in &map {
            used[usize::from(s & 7)] += 1;
        }
        Self {
            map,
            deltas,
            tree_probs: tree_probs(&used),
            lf: [0; 8],
            refs: [None; 8],
            skip: [false; 8],
            exact: false,
            seg_lambda: false,
        }
    }
}

/// Cyclic refresh (libvpx's `aq-mode=3`, `vp9_aq_cyclicrefresh.c`): each
/// inter frame codes `percent` of the picture's superblocks, the next
/// ones in raster order from `*next` (which moves on past them), in
/// segment 1 at a finer quantizer (`delta`), so that every part of the
/// picture is coded fine once every 100 / `percent` frames and what a
/// coarse frame left behind is cleaned up. The superblocks wrap around.
pub fn cyclic(
    next: &mut usize,
    (mi_cols, mi_rows): (usize, usize),
    percent: u32,
    delta: i16,
) -> Plan {
    let (sb_cols, sb_rows) = (mi_cols.div_ceil(8), mi_rows.div_ceil(8));
    let total = sb_cols * sb_rows;
    let count = (total * percent as usize).div_ceil(100).min(total);
    let mut map = vec![0u8; mi_cols * mi_rows];
    for k in 0..count {
        let sb = (*next + k) % total;
        let (r0, c0) = ((sb / sb_cols) * 8, (sb % sb_cols) * 8);
        for r in r0..(r0 + 8).min(mi_rows) {
            map[r * mi_cols + c0..r * mi_cols + (c0 + 8).min(mi_cols)].fill(1);
        }
    }
    *next = (*next + count) % total.max(1);
    let mut deltas = [0i16; 8];
    deltas[1] = delta;
    Plan::of_map(map, deltas)
}

/// The rate ratios of [`equator`]'s three bands (libvpx's `vp9_aq_360.c`
/// ratios of its segments 0 to 2).
pub const EQUATOR_RATIOS: [f64; 3] = [1.0, 0.75, 0.6];

/// Equirectangular (360°) pictures (libvpx's `aq-mode=4`): the rows of
/// 8×8 blocks away from the equator, which the projection stretches
/// across the width, in coarser segments, as libvpx's
/// `vp9_360aq_segment_id`: the eighth of the rows at the top and at the
/// bottom in segment 2, the next eighth on each side in 1, the half in
/// the middle in 0; `deltas` are the segments' quantizer deltas.
pub fn equator((mi_cols, mi_rows): (usize, usize), deltas: [i16; 8]) -> Plan {
    let mut map = vec![0u8; mi_cols * mi_rows];
    for r in 0..mi_rows {
        let seg = if r < mi_rows / 8 || r > mi_rows - mi_rows / 8 {
            2
        } else if r < mi_rows / 4 || r > mi_rows - mi_rows / 4 {
            1
        } else {
            0
        };
        map[r * mi_cols..(r + 1) * mi_cols].fill(seg);
    }
    Plan::of_map(map, deltas)
}

/// Complexity AQ (libvpx's `aq-mode=2`, `vp9_aq_complexity.c`): its
/// segment of the base quantizer, which the others' deltas move from.
pub const COMPLEXITY_NEUTRAL: usize = 3;

/// Complexity AQ by strength (libvpx's `get_aq_c_strength`, from the base
/// quantizer's step): each segment's rate over the base quantizer's
/// (`aq_c_q_adj_factor`), the most a superblock's projected rate may be
/// over the frame's mean for it (`aq_c_transitions`), and the most its
/// log variance may be over [`COMPLEXITY_LOW_VAR`] (`aq_c_var_thresholds`).
pub const COMPLEXITY_RATE: [[f64; 5]; 3] = [
    [1.75, 1.25, 1.05, 1.00, 0.90],
    [2.00, 1.50, 1.15, 1.00, 0.85],
    [2.50, 1.75, 1.25, 1.00, 0.80],
];
const COMPLEXITY_TRANSITIONS: [[f64; 5]; 3] = [
    [0.15, 0.30, 0.55, 2.00, 100.0],
    [0.20, 0.40, 0.65, 2.00, 100.0],
    [0.25, 0.50, 0.75, 2.00, 100.0],
];
const COMPLEXITY_VAR: [[f64; 5]; 3] = [
    [-4.0, -3.0, -2.0, 100.0, 100.0],
    [-3.5, -2.5, -1.5, 100.0, 100.0],
    [-3.0, -2.0, -1.0, 100.0, 100.0],
];

/// libvpx's `DEFAULT_LV_THRESH`: the log of 256 times a sample's variance
/// the thresholds of [`COMPLEXITY_VAR`] are from.
const COMPLEXITY_LOW_VAR: f64 = 10.0;

/// Complexity AQ's strength at the base quantizer's AC step `step` (8-bit
/// scale): libvpx's `get_aq_c_strength`.
pub fn complexity_strength(step: f64) -> usize {
    let base = (step / 4.0).trunc();
    usize::from(base > 10.0) + usize::from(base > 25.0)
}

/// Complexity AQ's map (libvpx's `vp9_caq_select_segment`, with the rate
/// projected from the source before the frame is coded): each
/// superblock's segment by its luma `y` (rows `stride` apart, `w`×`h`
/// visible, `bd` bits) against the base quantizer's AC step `step` (8-bit
/// scale): the first of segments 0 to 4 whose rate threshold its
/// projected rate a sample is under (as a share of the frame's mean) and
/// whose log variance threshold its log variance is under, so that flat,
/// cheap superblocks go finer and the costliest coarser. The projected
/// rate of each 8×8 block is the bits of a Gaussian of its variance at
/// the step in the pixels (an eighth of the coefficients'): half the log
/// of 1 + 12 times the variance over the step squared, a sample.
/// `deltas` are the segments' quantizer deltas.
#[allow(clippy::too_many_arguments)]
pub fn complexity<P: Pixel>(
    y: &[P],
    stride: usize,
    (w, h): (usize, usize),
    (mi_cols, mi_rows): (usize, usize),
    step: f64,
    strength: usize,
    bd: u8,
    deltas: [i16; 8],
) -> Plan {
    let (sc, sr) = (mi_cols.div_ceil(8), mi_rows.div_ceil(8));
    let scale = f64::from(1u32 << (2 * u32::from(bd - 8)));
    let q = (step / 8.0).max(1.0);
    // Each superblock's projected bits a sample and log variance.
    let mut sbs = Vec::with_capacity(sc * sr);
    let (mut bits, mut area) = (0.0f64, 0.0f64);
    for r in 0..sr {
        for c in 0..sc {
            let (x0, y0) = (c * 64, r * 64);
            let (x1, y1) = ((x0 + 64).min(w), (y0 + 64).min(h));
            let (mut sb_bits, mut n) = (0.0, 0usize);
            let (mut sum, mut sq) = (0f64, 0f64);
            for by in (y0..y1).step_by(8) {
                for bx in (x0..x1).step_by(8) {
                    let (bw, bh) = ((bx + 8).min(x1) - bx, (by + 8).min(y1) - by);
                    let (mut s1, mut s2) = (0f64, 0f64);
                    for yy in by..by + bh {
                        for &v in &y[yy * stride + bx..yy * stride + bx + bw] {
                            let v = f64::from(v.get());
                            s1 += v;
                            s2 += v * v;
                        }
                    }
                    let k = (bw * bh) as f64;
                    let var = ((s2 - s1 * s1 / k) / k / scale).max(0.0);
                    sb_bits += k * 0.5 * (1.0 + 12.0 * var / (q * q)).log2();
                    sum += s1;
                    sq += s2;
                    n += bw * bh;
                }
            }
            let k = n.max(1) as f64;
            let var = ((sq - sum * sum / k) / k / scale).max(0.0);
            sbs.push((sb_bits / k, (256.0 * var + 1.0).ln()));
            bits += sb_bits;
            area += k;
        }
    }
    let mean = bits / area.max(1.0);
    let t = &COMPLEXITY_TRANSITIONS[strength];
    let v = &COMPLEXITY_VAR[strength];
    let mut map = vec![0u8; mi_cols * mi_rows];
    for (i, &(rate, logvar)) in sbs.iter().enumerate() {
        let seg = (0..5)
            .find(|&k| rate < mean * t[k] && logvar < COMPLEXITY_LOW_VAR + v[k])
            .unwrap_or(4) as u8;
        let (r0, c0) = ((i / sc) * 8, (i % sc) * 8);
        for r in r0..(r0 + 8).min(mi_rows) {
            map[r * mi_cols + c0..r * mi_cols + (c0 + 8).min(mi_cols)].fill(seg);
        }
    }
    let mut plan = Plan::of_map(map, deltas);
    plan.seg_lambda = true;
    plan
}

/// The segment tree's probabilities for ids used `used` times.
fn tree_probs(used: &[u32; 8]) -> [u8; 7] {
    let mut branch = [[0u64; 2]; 7];
    for (seg, &n) in used.iter().enumerate() {
        super::probs::walk(&prob::SEG_TREE, seg as u8, |bit, node| {
            branch[node][bit] += u64::from(n);
        });
    }
    branch.map(|[a, b]| {
        let t = a + b;
        (256 * a + t / 2)
            .checked_div(t)
            .map_or(255, |p| p.clamp(1, 255) as u8)
    })
}
