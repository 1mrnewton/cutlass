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
