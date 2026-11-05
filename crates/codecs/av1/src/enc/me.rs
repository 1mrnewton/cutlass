//! Motion estimation and distortion kernels for the encoder.
//!
//! # Distortion kernels
//!
//! [`sad`], [`sse`] and [`satd`] compare two `w` x `h` blocks of [`Pixel`]s
//! given as slices with a stride in samples. Every AV1 block shape is
//! supported (`w`, `h` in `4..=128`, including 4x16, 16x4, 8x32, ...). The
//! common widths (4, 8, 16, 32, 64, 128) are specialised so the row loops have
//! a constant length and vectorise. A slice holding a block needs at least
//! `(h - 1) * stride + w` samples.
//!
//! [`block_sum`], [`block_mean`], [`block_variance`] and [`diff_variance`]
//! serve adaptive quantisation and mode decisions; [`sse_i16`], [`sse_i32`]
//! and [`energy_i32`] measure residual and coefficient buffers.
//!
//! # Motion search
//!
//! [`motion_search`] finds a luma motion vector (1/8 sample units, `(row,
//! col)`, as coded by AV1) for one block: it evaluates the predicted
//! candidates, runs an iterative square-pattern step search with decreasing
//! steps (SAD + rate, with early termination), then refines to half, quarter
//! and eighth sample positions with the real AV1 interpolation filters
//! ([`put`]) and [`satd`]. All working memory lives in a reusable
//! [`MeScratch`], so a search allocates nothing.
//!
//! [`downscale_2x`] builds half-resolution planes for a coarse-to-fine
//! search: search the downscaled planes with the halved block, then pass the
//! doubled vector as a candidate at full resolution.

use crate::dsp::Pixel;
use crate::dsp::mc::{emu_edge, put, put_h_rows, put_v_rows};

// ---------------------------------------------------------------------------
// SAD / SSE
// ---------------------------------------------------------------------------

/// Sum of absolute differences of rows, stopping after the first row whose
/// running sum exceeds `limit`. Inlined into width-specialised callers so
/// `w` is a constant there.
#[inline(always)]
fn sad_rows<P: Pixel>(
    a: &[P],
    a_stride: usize,
    b: &[P],
    b_stride: usize,
    w: usize,
    h: usize,
    limit: u32,
) -> u32 {
    let mut sum = 0u32;
    for (ra, rb) in a.chunks(a_stride).zip(b.chunks(b_stride)).take(h) {
        let mut s = 0u32;
        for (&x, &y) in ra[..w].iter().zip(&rb[..w]) {
            s += (x.to_i32() - y.to_i32()).unsigned_abs();
        }
        sum += s;
        if sum > limit {
            break;
        }
    }
    sum
}

/// [`sad`] that may stop early: once the partial sum exceeds `limit` it
/// returns that partial sum (a value `> limit`). The result is exact whenever
/// it is `<= limit`.
pub fn sad_limited<P: Pixel>(
    a: &[P],
    a_stride: usize,
    b: &[P],
    b_stride: usize,
    w: usize,
    h: usize,
    limit: u32,
) -> u32 {
    match w {
        4 => sad_rows(a, a_stride, b, b_stride, 4, h, limit),
        8 => sad_rows(a, a_stride, b, b_stride, 8, h, limit),
        16 => sad_rows(a, a_stride, b, b_stride, 16, h, limit),
        32 => sad_rows(a, a_stride, b, b_stride, 32, h, limit),
        64 => sad_rows(a, a_stride, b, b_stride, 64, h, limit),
        128 => sad_rows(a, a_stride, b, b_stride, 128, h, limit),
        _ => sad_rows(a, a_stride, b, b_stride, w, h, limit),
    }
}

/// Sum of absolute differences of two `w` x `h` blocks.
pub fn sad<P: Pixel>(
    a: &[P],
    a_stride: usize,
    b: &[P],
    b_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    sad_limited(a, a_stride, b, b_stride, w, h, u32::MAX)
}

/// Row-wise SSE with a `u32` row accumulator (exact for `w <= 128` at 12
/// bits: `128 * 4095^2 < 2^32`).
#[inline(always)]
fn sse_rows<P: Pixel>(
    a: &[P],
    a_stride: usize,
    b: &[P],
    b_stride: usize,
    w: usize,
    h: usize,
) -> u64 {
    let mut sum = 0u64;
    for (ra, rb) in a.chunks(a_stride).zip(b.chunks(b_stride)).take(h) {
        let mut s = 0u32;
        for (&x, &y) in ra[..w].iter().zip(&rb[..w]) {
            let d = x.to_i32() - y.to_i32();
            s += (d * d) as u32;
        }
        sum += u64::from(s);
    }
    sum
}

/// Row-wise SSE for arbitrary widths.
fn sse_rows_wide<P: Pixel>(
    a: &[P],
    a_stride: usize,
    b: &[P],
    b_stride: usize,
    w: usize,
    h: usize,
) -> u64 {
    let mut sum = 0u64;
    for (ra, rb) in a.chunks(a_stride).zip(b.chunks(b_stride)).take(h) {
        for (&x, &y) in ra[..w].iter().zip(&rb[..w]) {
            let d = i64::from(x.to_i32() - y.to_i32());
            sum += (d * d) as u64;
        }
    }
    sum
}

/// Sum of squared differences of two `w` x `h` blocks.
pub fn sse<P: Pixel>(
    a: &[P],
    a_stride: usize,
    b: &[P],
    b_stride: usize,
    w: usize,
    h: usize,
) -> u64 {
    match w {
        4 => sse_rows(a, a_stride, b, b_stride, 4, h),
        8 => sse_rows(a, a_stride, b, b_stride, 8, h),
        16 => sse_rows(a, a_stride, b, b_stride, 16, h),
        32 => sse_rows(a, a_stride, b, b_stride, 32, h),
        64 => sse_rows(a, a_stride, b, b_stride, 64, h),
        128 => sse_rows(a, a_stride, b, b_stride, 128, h),
        _ if w <= 128 => sse_rows(a, a_stride, b, b_stride, w, h),
        _ => sse_rows_wide(a, a_stride, b, b_stride, w, h),
    }
}

/// Sum of squared differences of two equally long `i16` buffers (residuals).
pub fn sse_i16(a: &[i16], b: &[i16]) -> u64 {
    let mut sum = 0u64;
    for (ca, cb) in a.chunks(64).zip(b.chunks(64)) {
        // |d| <= 65535, d^2 < 2^32: accumulate a chunk in u64 lanes.
        let mut s = 0u64;
        for (&x, &y) in ca.iter().zip(cb) {
            let d = i64::from(x) - i64::from(y);
            s += (d * d) as u64;
        }
        sum += s;
    }
    sum
}

/// Sum of squared differences of two equally long `i32` buffers
/// (coefficients). Values must fit in 31 bits of magnitude difference.
pub fn sse_i32(a: &[i32], b: &[i32]) -> u64 {
    let mut sum = 0u64;
    for (&x, &y) in a.iter().zip(b) {
        let d = i64::from(x) - i64::from(y);
        sum = sum.wrapping_add((d * d) as u64);
    }
    sum
}

/// Sum of squares of an `i32` buffer.
pub fn energy_i32(a: &[i32]) -> u64 {
    let mut sum = 0u64;
    for &x in a {
        let v = i64::from(x);
        sum = sum.wrapping_add((v * v) as u64);
    }
    sum
}

// ---------------------------------------------------------------------------
// Mean / variance
// ---------------------------------------------------------------------------

#[inline(always)]
fn sum_rows<P: Pixel>(a: &[P], stride: usize, w: usize, h: usize) -> (u32, u64) {
    let mut sum = 0u32;
    let mut sq = 0u64;
    for r in a.chunks(stride).take(h) {
        let mut s = 0u32;
        let mut q = 0u32;
        for &x in &r[..w] {
            let v = x.to_i32() as u32;
            s += v;
            q += v * v;
        }
        sum += s;
        sq += u64::from(q);
    }
    (sum, sq)
}

/// Sum and sum of squares of a `w` x `h` block (`w <= 128`).
pub fn block_sum<P: Pixel>(a: &[P], stride: usize, w: usize, h: usize) -> (u32, u64) {
    match w {
        4 => sum_rows(a, stride, 4, h),
        8 => sum_rows(a, stride, 8, h),
        16 => sum_rows(a, stride, 16, h),
        32 => sum_rows(a, stride, 32, h),
        64 => sum_rows(a, stride, 64, h),
        128 => sum_rows(a, stride, 128, h),
        _ => sum_rows(a, stride, w, h),
    }
}

/// Mean sample value of a block, rounded to nearest.
pub fn block_mean<P: Pixel>(a: &[P], stride: usize, w: usize, h: usize) -> u32 {
    let n = (w * h) as u64;
    if n == 0 {
        return 0;
    }
    let (s, _) = block_sum(a, stride, w, h);
    ((u64::from(s) + n / 2) / n) as u32
}

/// Sum of squared deviations from the block mean,
/// `sum(x^2) - sum(x)^2 / n` (floor). Divide by `w * h` for the variance.
pub fn block_variance<P: Pixel>(a: &[P], stride: usize, w: usize, h: usize) -> u64 {
    let n = (w * h) as u64;
    if n == 0 {
        return 0;
    }
    let (s, q) = block_sum(a, stride, w, h);
    let s = u64::from(s);
    q - s * s / n
}

#[inline(always)]
fn diff_rows<P: Pixel>(
    a: &[P],
    a_stride: usize,
    b: &[P],
    b_stride: usize,
    w: usize,
    h: usize,
) -> (i64, u64) {
    let mut sum = 0i64;
    let mut sq = 0u64;
    for (ra, rb) in a.chunks(a_stride).zip(b.chunks(b_stride)).take(h) {
        let mut s = 0i32;
        let mut q = 0u32;
        for (&x, &y) in ra[..w].iter().zip(&rb[..w]) {
            let d = x.to_i32() - y.to_i32();
            s += d;
            q += (d * d) as u32;
        }
        sum += i64::from(s);
        sq += u64::from(q);
    }
    (sum, sq)
}

/// SSE and variance of the difference `a - b` (`w <= 128`): returns
/// `(sse, sse - sum(a - b)^2 / n)`, the second being the SSE left after
/// removing the mean difference (a DC-blind distortion).
pub fn diff_variance<P: Pixel>(
    a: &[P],
    a_stride: usize,
    b: &[P],
    b_stride: usize,
    w: usize,
    h: usize,
) -> (u64, u64) {
    let n = (w * h) as u64;
    if n == 0 {
        return (0, 0);
    }
    let (s, q) = match w {
        4 => diff_rows(a, a_stride, b, b_stride, 4, h),
        8 => diff_rows(a, a_stride, b, b_stride, 8, h),
        16 => diff_rows(a, a_stride, b, b_stride, 16, h),
        32 => diff_rows(a, a_stride, b, b_stride, 32, h),
        64 => diff_rows(a, a_stride, b, b_stride, 64, h),
        128 => diff_rows(a, a_stride, b, b_stride, 128, h),
        _ => diff_rows(a, a_stride, b, b_stride, w, h),
    };
    let s2 = s.unsigned_abs() * s.unsigned_abs();
    (q, q - s2 / n)
}

// ---------------------------------------------------------------------------
// SATD
// ---------------------------------------------------------------------------

/// In-place unnormalised Walsh-Hadamard transform along the first index
/// (butterflies across rows, element-wise over a row so they vectorise).
#[inline(always)]
fn wht_cols<const N: usize>(m: &mut [[i32; N]; N]) {
    let mut half = 1;
    while half < N {
        let mut i = 0;
        while i < N {
            for j in i..i + half {
                let (lo, hi) = m.split_at_mut(j + half);
                let (a, b) = (&mut lo[j], &mut hi[0]);
                for k in 0..N {
                    let (x, y) = (a[k], b[k]);
                    a[k] = x + y;
                    b[k] = x - y;
                }
            }
            i += 2 * half;
        }
        half *= 2;
    }
}

#[inline(always)]
fn transpose<const N: usize>(m: &[[i32; N]; N]) -> [[i32; N]; N] {
    let mut t = [[0i32; N]; N];
    for (i, row) in m.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            t[j][i] = v;
        }
    }
    t
}

/// Sum of absolute 2-D Hadamard coefficients of the `N` x `N` difference
/// block, unnormalised.
#[inline(always)]
fn hadamard_abs<P: Pixel, const N: usize>(
    a: &[P],
    a_stride: usize,
    b: &[P],
    b_stride: usize,
) -> u32 {
    let mut m = [[0i32; N]; N];
    for (row, (ra, rb)) in m.iter_mut().zip(a.chunks(a_stride).zip(b.chunks(b_stride))) {
        for (d, (&x, &y)) in row.iter_mut().zip(ra[..N].iter().zip(&rb[..N])) {
            *d = x.to_i32() - y.to_i32();
        }
    }
    wht_cols(&mut m);
    let mut t = transpose(&m);
    wht_cols(&mut t);
    let mut s = 0u32;
    for row in &t {
        for &v in row {
            s += v.unsigned_abs();
        }
    }
    s
}

/// One butterfly stage inside an 8-lane row: lane `k` pairs with
/// `v[perm[k]]` and receives the sum (`sign[k] = 1`) or the difference
/// (`sign[k] = -1`). Single-source lane permutations become shuffles.
#[inline(always)]
fn row_stage(v: [i16; 8], perm: [usize; 8], sign: [i16; 8]) -> [i16; 8] {
    core::array::from_fn(|k| v[perm[k]].wrapping_add(v[k].wrapping_mul(sign[k])))
}

/// [`hadamard_abs`] of an 8x8 block of 8-bit samples in 16-bit lanes, one
/// row per vector: the row-index stages across rows, two column-index
/// stages by lane permutations and the last one as pairwise sums and
/// differences widened to 32 bits. Differences lie in `-255..=255`, so every
/// 16-bit intermediate stays within `32 * 255`.
#[inline(always)]
fn hadamard8_abs_8bit<P: Pixel>(a: &[P], a_stride: usize, b: &[P], b_stride: usize) -> u32 {
    let a = &a[..7 * a_stride + 8];
    let b = &b[..7 * b_stride + 8];
    let mut m = [[0i16; 8]; 8];
    for (r, row) in m.iter_mut().enumerate() {
        let ra = &a[r * a_stride..][..8];
        let rb = &b[r * b_stride..][..8];
        *row = core::array::from_fn(|k| (ra[k].to_i32() - rb[k].to_i32()) as i16);
    }
    for half in [1, 2, 4] {
        let mut i = 0;
        while i < 8 {
            for j in i..i + half {
                let (x, y) = (m[j], m[j + half]);
                m[j] = core::array::from_fn(|k| x[k].wrapping_add(y[k]));
                m[j + half] = core::array::from_fn(|k| x[k].wrapping_sub(y[k]));
            }
            i += 2 * half;
        }
    }
    let mut acc = [0i32; 4];
    for row in &m {
        let v = row_stage(*row, [4, 5, 6, 7, 0, 1, 2, 3], [1, 1, 1, 1, -1, -1, -1, -1]);
        let v = row_stage(v, [2, 3, 0, 1, 6, 7, 4, 5], [1, 1, -1, -1, 1, 1, -1, -1]);
        for k in 0..4 {
            let (x, y) = (i32::from(v[2 * k]), i32::from(v[2 * k + 1]));
            acc[k] += (x + y).abs() + (x - y).abs();
        }
    }
    acc.iter().sum::<i32>() as u32
}

#[inline(always)]
fn satd_tiles<P: Pixel, const N: usize>(
    a: &[P],
    a_stride: usize,
    b: &[P],
    b_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    let mut total = 0u32;
    for by in (0..h).step_by(N) {
        for bx in (0..w).step_by(N) {
            let (ta, tb) = (&a[by * a_stride + bx..], &b[by * b_stride + bx..]);
            let s = if N == 8 && !P::HBD {
                hadamard8_abs_8bit(ta, a_stride, tb, b_stride)
            } else {
                hadamard_abs::<P, N>(ta, a_stride, tb, b_stride)
            };
            total += (s + 1) >> 1;
        }
    }
    total
}

/// Sum of absolute Hadamard-transformed differences.
///
/// The `w` x `h` block (`w`, `h` multiples of 4) is tiled with 8x8
/// transforms when both dimensions are multiples of 8, otherwise with 4x4
/// transforms. Each tile contributes `(S + 1) >> 1`, where `S` is the sum of
/// absolute values of the unnormalised 2-D Walsh-Hadamard transform of the
/// difference `a - b`. Both tile sizes are on the SAD scale: a flat
/// difference `d` gives `|d| * w * h / 2`, half its SAD.
pub fn satd<P: Pixel>(
    a: &[P],
    a_stride: usize,
    b: &[P],
    b_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    if w.is_multiple_of(8) && h.is_multiple_of(8) {
        satd_tiles::<P, 8>(a, a_stride, b, b_stride, w, h)
    } else {
        satd_tiles::<P, 4>(a, a_stride, b, b_stride, w, h)
    }
}

// ---------------------------------------------------------------------------
// Downscaling
// ---------------------------------------------------------------------------

/// Dimensions of the plane [`downscale_2x`] produces from a `w` x `h` plane.
pub fn downscale_2x_dims(w: usize, h: usize) -> (usize, usize) {
    (w.div_ceil(2), h.div_ceil(2))
}

/// Halves a `w` x `h` plane in both directions: each output sample is the
/// rounded mean of a 2x2 input square (the last row / column is repeated
/// for odd sizes). `dst` holds `downscale_2x_dims(w, h)` samples with stride
/// `dst_stride`. Apply twice for 4x.
pub fn downscale_2x<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    src: &[P],
    src_stride: usize,
    w: usize,
    h: usize,
) {
    if w == 0 || h == 0 {
        return;
    }
    let (ow, oh) = downscale_2x_dims(w, h);
    let pairs = w / 2;
    for r in 0..oh {
        let r0 = &src[2 * r * src_stride..][..w];
        let r1 = &src[(2 * r + 1).min(h - 1) * src_stride..][..w];
        let d = &mut dst[r * dst_stride..][..ow];
        for ((o, p0), p1) in d[..pairs]
            .iter_mut()
            .zip(r0.chunks_exact(2))
            .zip(r1.chunks_exact(2))
        {
            let s = p0[0].to_i32() + p0[1].to_i32() + p1[0].to_i32() + p1[1].to_i32();
            *o = P::from_i32((s + 2) >> 2);
        }
        if w % 2 == 1 {
            let s = 2 * (r0[w - 1].to_i32() + r1[w - 1].to_i32());
            d[ow - 1] = P::from_i32((s + 2) >> 2);
        }
    }
}

// ---------------------------------------------------------------------------
// Motion search
// ---------------------------------------------------------------------------

/// Largest coded MV component magnitude in 1/8 samples (the spec requires
/// `MV_LOW < mv < MV_UPP` with `MV_UPP = 1 << 14`).
pub const MV_MAX: i32 = (1 << 14) - 1;

/// Largest block side.
const MAX_BLOCK: usize = 128;
/// Edge-emulation buffer side: the block plus the 8-tap footprint.
const EDGE_SIDE: usize = MAX_BLOCK + 7;
/// Full-sample offset around the best candidate that an edge-emulated
/// search window covers.
const WIN_REACH: i32 = 24;

/// Motion search settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeParams {
    /// Full-sample search radius around the best predicted candidate (each
    /// component). 0 evaluates the candidates only.
    pub max_range: i32,
    /// Sub-sample refinement: 0 full sample only, 1 half, 2 quarter,
    /// 3 eighth (needs `allow_high_precision_mv`).
    pub subpel: u8,
    /// Interpolation filter used to evaluate sub-sample positions
    /// (0 regular, 1 smooth, 2 sharp, 3 bilinear). The 4-tap variant is
    /// substituted for a dimension of at most 4, as the decoder does.
    pub filter: u8,
    /// How far, in samples, the predicted block may lie beyond the frame:
    /// the block's top-left column stays in `-(w + border)..=ref_w + border`
    /// (rows likewise). AV1 itself allows any position; past the edge the
    /// prediction only repeats edge samples.
    pub border: i32,
    /// Moves allowed per step size of the full-sample pattern search.
    pub max_iters: u32,
    /// Largest ring radius the full-sample search starts with.
    pub max_ring: i32,
    /// Sub-sample steps try the four axis neighbours and then only the
    /// diagonal between the better ones (5 predictions instead of 8).
    pub subpel_pruned: bool,
}

impl Default for MeParams {
    fn default() -> Self {
        Self {
            max_range: 64,
            subpel: 2,
            filter: 0,
            border: 16,
            max_iters: 8,
            max_ring: 16,
            subpel_pruned: false,
        }
    }
}

/// Outcome of [`motion_search`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeResult {
    /// Motion vector in 1/8 luma samples, `(row, col)`.
    pub mv: (i32, i32),
    /// `dist + mv_cost(mv)`.
    pub cost: u32,
    /// Distortion of `mv`: [`satd`] when sub-sample refinement ran
    /// (`subpel > 0`), [`sad`] otherwise.
    pub dist: u32,
}

/// Reusable working memory for [`motion_search`] (blocks up to 128x128).
#[derive(Clone, Debug)]
pub struct MeScratch<P: Pixel> {
    /// Interpolated prediction, stride = block width.
    pred: Vec<P>,
    /// Edge-emulated reference footprint.
    edge: Vec<P>,
    /// Partition estimates of the current superblock: `[x, y, w, h]`, cost
    /// and motion (kept here for the mode decision, which owns this scratch).
    pub est_memo: Vec<([usize; 4], f64, (i32, i32))>,
    /// The partition the next `enc_partition` of this node (row, column,
    /// size) returns.
    pub force_part: Option<(usize, usize, u8, usize)>,
    /// Edge-emulated reference area of a pattern search that reaches past the plane.
    win: Vec<P>,
    /// The mode search's recent predictions of the current block.
    pub pred_cache: Vec<(super::rdo::PredKey, Vec<P>)>,
    /// The transform cost of the chosen inter prediction and transform size.
    pub tx_memo: Option<(super::rdo::PredKey, u8, f64)>,
    /// Intermediate of the estimate transform.
    pub tx_tmp: Vec<i16>,
    /// Horizontal-pass rows the sub-sample search shares.
    cols: ColRows,
}

/// One reference column and horizontal phase of the sub-sample search:
/// its horizontal-pass rows (see [`put_h_rows`]), held for source rows
/// `lo..hi` with buffer row 0 at source row `row0`.
#[derive(Clone, Debug, Default)]
struct ColSlot {
    /// `(ix, mx)`, or `None` when unused.
    key: Option<(isize, usize)>,
    row0: isize,
    lo: isize,
    hi: isize,
    mid: Vec<i16>,
    high: Vec<i16>,
}

/// Rows of the sub-sample candidates of one search that share a column
/// and horizontal phase (the predictions above and below each other), so
/// each row is filtered horizontally once.
#[derive(Clone, Debug, Default)]
struct ColRows {
    slots: [ColSlot; 3],
    next: usize,
}

/// Rows a [`ColSlot`] holds beyond those of one prediction, above and
/// below.
const COL_PAD: usize = 2;

impl ColRows {
    fn clear(&mut self) {
        for s in &mut self.slots {
            s.key = None;
        }
    }
}

impl<P: Pixel> MeScratch<P> {
    pub fn new() -> Self {
        Self {
            cols: ColRows::default(),
            pred: vec![P::default(); MAX_BLOCK * MAX_BLOCK],
            edge: vec![P::default(); EDGE_SIDE * EDGE_SIDE],
            est_memo: Vec::with_capacity(128),
            force_part: None,
            win: Vec::new(),
            pred_cache: Vec::new(),
            tx_memo: None,
            tx_tmp: vec![0; 64 * 32],
        }
    }

    /// A placeholder that allocates nothing (decoding never searches).
    pub fn empty() -> Self {
        Self {
            cols: ColRows::default(),
            pred: Vec::new(),
            edge: Vec::new(),
            est_memo: Vec::new(),
            force_part: None,
            win: Vec::new(),
            pred_cache: Vec::new(),
            tx_memo: None,
            tx_tmp: Vec::new(),
        }
    }
}

impl<P: Pixel> Default for MeScratch<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// The reference plane and the block being searched.
struct Search<'a, P: Pixel> {
    src: &'a [P],
    src_stride: usize,
    x: isize,
    y: isize,
    w: usize,
    h: usize,
    refp: &'a [P],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    mv_cost: &'a dyn Fn(i32, i32) -> u32,
    bd: u32,
    filter_x: usize,
    filter_y: usize,
    /// An edge-emulated copy of the reference area the pattern search
    /// covers when it reaches past the plane (`win_w == 0`: none), with its
    /// top-left plane position.
    win: &'a [P],
    win_x: isize,
    win_y: isize,
    win_w: usize,
    win_h: usize,
}

/// Running best full-sample offset (samples) or MV (1/8 samples).
#[derive(Clone, Copy)]
struct Best {
    row: i32,
    col: i32,
    cost: u32,
    dist: u32,
}

const SQUARE: [(i32, i32); 8] = [
    (-1, 0),
    (0, -1),
    (0, 1),
    (1, 0),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
];

/// The 4-tap filter row used for a block dimension of at most 4.
fn four_tap(filter: u8) -> usize {
    match filter {
        1 | 5 => 5,
        3 => 3,
        _ => 4,
    }
}

impl<P: Pixel> Search<'_, P> {
    fn src_block(&self) -> &[P] {
        &self.src[self.y as usize * self.src_stride + self.x as usize..]
    }

    /// Index in `win` of the `w` x `h` area at plane position `(px, py)`,
    /// when the window holds all of it.
    fn in_win(&self, px: isize, py: isize, w: usize, h: usize) -> Option<usize> {
        let (dx, dy) = (px - self.win_x, py - self.win_y);
        (self.win_w > 0
            && dx >= 0
            && dy >= 0
            && dx as usize + w <= self.win_w
            && dy as usize + h <= self.win_h)
            .then(|| dy as usize * self.win_w + dx as usize)
    }

    /// The reference block at full-sample offset `(dy, dx)`: directly from
    /// the plane when inside, else edge-emulated into `edge` (stride `w`).
    fn ref_block<'b>(&'b self, edge: &'b mut [P], dy: i32, dx: i32) -> (&'b [P], usize) {
        let px = self.x + dx as isize;
        let py = self.y + dy as isize;
        if px >= 0
            && py >= 0
            && px as usize + self.w <= self.ref_w
            && py as usize + self.h <= self.ref_h
        {
            (
                &self.refp[py as usize * self.ref_stride + px as usize..],
                self.ref_stride,
            )
        } else if let Some(o) = self.in_win(px, py, self.w, self.h) {
            (&self.win[o..], self.win_w)
        } else {
            emu_edge(
                edge,
                self.w,
                self.refp,
                self.ref_stride,
                self.ref_w,
                self.ref_h,
                px,
                py,
                self.w,
                self.h,
            );
            (&edge[..], self.w)
        }
    }

    /// SAD + rate at full-sample offset `(dy, dx)`, or `None` when it cannot
    /// beat `best_cost`.
    fn eval_full(&self, edge: &mut [P], dy: i32, dx: i32, best_cost: u32) -> Option<(u32, u32)> {
        let rate = (self.mv_cost)(dy * 8, dx * 8);
        if rate >= best_cost {
            return None;
        }
        let limit = best_cost - rate;
        let (blk, bs) = self.ref_block(edge, dy, dx);
        let d = sad_limited(
            self.src_block(),
            self.src_stride,
            blk,
            bs,
            self.w,
            self.h,
            limit,
        );
        if d >= limit {
            None
        } else {
            Some((d + rate, d))
        }
    }

    /// [`Self::predict`], filtering each source row horizontally once for
    /// the predictions in one column and phase (those inside the plane
    /// with both fractions nonzero).
    fn predict_shared(&self, cols: &mut ColRows, pred: &mut [P], edge: &mut [P], mv: (i32, i32)) {
        let py = self.y * 8 + mv.0 as isize;
        let px = self.x * 8 + mv.1 as isize;
        let (iy, ix) = (py >> 3, px >> 3);
        let my = ((py & 7) * 2) as usize;
        let mx = ((px & 7) * 2) as usize;
        let (w, h) = (self.w, self.h);
        if mx == 0
            || my == 0
            || ix < 3
            || iy < 3
            || ix as usize + w + 4 > self.ref_w
            || iy as usize + h + 4 > self.ref_h
        {
            return self.predict(pred, edge, mv);
        }
        let cap = h + 7 + 2 * COL_PAD;
        let (lo, hi) = (iy - 3, iy + h as isize + 4);
        let key = Some((ix, mx));
        let k = match cols
            .slots
            .iter()
            .position(|s| s.key == key && lo >= s.row0 && hi <= s.row0 + cap as isize)
        {
            Some(k) => k,
            None => {
                let k = cols.next;
                cols.next = (k + 1) % cols.slots.len();
                let s = &mut cols.slots[k];
                s.key = key;
                s.row0 = lo - COL_PAD as isize;
                s.lo = lo;
                s.hi = lo;
                if s.mid.len() < cap * w {
                    s.mid.resize(cap * w, 0);
                    s.high.resize(cap * w, 0);
                }
                k
            }
        };
        let s = &mut cols.slots[k];
        let fill = |a: isize, b: isize, s: &mut ColSlot| {
            let o = (a - s.row0) as usize * w;
            put_h_rows(
                &mut s.mid[o..],
                &mut s.high[o..],
                self.refp,
                a as usize * self.ref_stride + ix as usize - 3,
                self.ref_stride,
                w,
                (b - a) as usize,
                mx,
                self.filter_x,
                self.bd,
            );
        };
        if s.lo == s.hi {
            fill(lo, hi, s);
            (s.lo, s.hi) = (lo, hi);
        } else {
            if lo < s.lo {
                fill(lo, s.lo, s);
                s.lo = lo;
            }
            if hi > s.hi {
                fill(s.hi, hi, s);
                s.hi = hi;
            }
        }
        put_v_rows(
            pred,
            w,
            &s.mid,
            &s.high,
            (lo - s.row0) as usize * w,
            w,
            h,
            my,
            self.filter_y,
            self.bd,
        );
    }

    /// Interpolates the prediction for `mv` (1/8 samples) into `pred`
    /// (stride `w`).
    fn predict(&self, pred: &mut [P], edge: &mut [P], mv: (i32, i32)) {
        let py = self.y * 8 + mv.0 as isize;
        let px = self.x * 8 + mv.1 as isize;
        let (iy, ix) = (py >> 3, px >> 3);
        let my = ((py & 7) * 2) as usize;
        let mx = ((px & 7) * 2) as usize;
        let (w, h) = (self.w, self.h);
        if ix >= 3
            && iy >= 3
            && ix as usize + w + 4 <= self.ref_w
            && iy as usize + h + 4 <= self.ref_h
        {
            put(
                pred,
                w,
                self.refp,
                iy as usize * self.ref_stride + ix as usize,
                self.ref_stride,
                w,
                h,
                mx,
                my,
                self.filter_x,
                self.filter_y,
                self.bd,
            );
        } else if let Some(o) = self.in_win(ix - 3, iy - 3, w + 7, h + 7) {
            let ws = self.win_w;
            put(
                pred,
                w,
                self.win,
                o + 3 * ws + 3,
                ws,
                w,
                h,
                mx,
                my,
                self.filter_x,
                self.filter_y,
                self.bd,
            );
        } else {
            let es = w + 7;
            emu_edge(
                edge,
                es,
                self.refp,
                self.ref_stride,
                self.ref_w,
                self.ref_h,
                ix - 3,
                iy - 3,
                es,
                h + 7,
            );
            put(
                pred,
                w,
                edge,
                3 * es + 3,
                es,
                w,
                h,
                mx,
                my,
                self.filter_x,
                self.filter_y,
                self.bd,
            );
        }
    }
}

/// Rounds a 1/8-sample component to the nearest full sample.
fn round_full(v: i32) -> i32 {
    (v + 4) >> 3
}

/// Searches `refp` (a `ref_w` x `ref_h` plane with stride `ref_stride`) for
/// the `w` x `h` block of `src` at plane position `(x, y)`.
///
/// * `candidates` are predicted MVs (1/8 samples, `(row, col)`) such as the
///   AV1 MV stack, neighbours, or a doubled coarse-level vector; the zero MV
///   is always tried first. Each is rounded to full samples.
/// * `mv_cost(row, col)` returns the rate term (already scaled by lambda)
///   of a 1/8-sample MV.
///
/// Steps: (1) candidates by SAD + rate; (2) from the best one, 8-point
/// rings of radius 1, 2, 4, .. up to `min(pow2 <= max_range, 16)`, then an
/// iterative 8-neighbour pattern search with steps from half the winning
/// ring radius down to 1 (at most `max_iters` moves per step), all confined
/// to `best_candidate +- max_range`; (3) if `subpel > 0`, 8-neighbour
/// refinement at 1/2, 1/4, 1/8 steps by SATD + rate on the AV1-interpolated
/// prediction. Every evaluated position satisfies the [`MeParams::border`]
/// window and `|mv| <= MV_MAX`. Ties keep the earlier position.
///
/// `w`, `h` in `4..=128`; `bd` is the bit depth (8, 10, 12). The block must
/// lie inside `src`.
pub fn motion_search<P: Pixel>(
    scratch: &mut MeScratch<P>,
    src: &[P],
    src_stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    refp: &[P],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    candidates: &[(i32, i32)],
    params: &MeParams,
    mv_cost: &dyn Fn(i32, i32) -> u32,
    bd: u32,
) -> MeResult {
    let filter = usize::from(params.filter);
    let s = Search {
        src,
        src_stride,
        x: x as isize,
        y: y as isize,
        w,
        h,
        refp,
        ref_stride,
        ref_w,
        ref_h,
        mv_cost,
        bd,
        filter_x: if w <= 4 {
            four_tap(params.filter)
        } else {
            filter
        },
        filter_y: if h <= 4 {
            four_tap(params.filter)
        } else {
            filter
        },
        win: &[],
        win_x: 0,
        win_y: 0,
        win_w: 0,
        win_h: 0,
    };
    let MeScratch {
        pred,
        edge,
        win,
        cols,
        ..
    } = scratch;

    // Full-sample window (offsets in samples).
    let full_max = MV_MAX >> 3;
    let border = params.border.max(0);
    let (xi, yi) = (x as i32, y as i32);
    let col_lo = (-xi - w as i32 - border).max(-full_max);
    let col_hi = (ref_w as i32 + border - xi).min(full_max).max(col_lo);
    let row_lo = (-yi - h as i32 - border).max(-full_max);
    let row_hi = (ref_h as i32 + border - yi).min(full_max).max(row_lo);

    // (1) Candidates.
    let zero = (0.clamp(row_lo, row_hi), 0.clamp(col_lo, col_hi));
    let mut best = Best {
        row: zero.0,
        col: zero.1,
        cost: u32::MAX,
        dist: u32::MAX,
    };
    if let Some((c, d)) = s.eval_full(edge, zero.0, zero.1, u32::MAX) {
        best.cost = c;
        best.dist = d;
    }
    let full = |c: &(i32, i32)| {
        (
            round_full(c.0).clamp(row_lo, row_hi),
            round_full(c.1).clamp(col_lo, col_hi),
        )
    };
    for (i, c) in candidates.iter().enumerate() {
        let (dy, dx) = full(c);
        if (dy, dx) == zero || candidates[..i].iter().any(|p| full(p) == (dy, dx)) {
            continue;
        }
        if let Some((c, d)) = s.eval_full(edge, dy, dx, best.cost) {
            best = Best {
                row: dy,
                col: dx,
                cost: c,
                dist: d,
            };
        }
    }

    // (2) Pattern search around the best candidate. When the area it is
    // likely to cover (plus the interpolation taps) reaches past the plane,
    // that area is edge-emulated once instead of block by block.
    let range = params.max_range.max(0);
    let reach = range.min(WIN_REACH) as isize;
    let (wx0, wy0) = (
        x as isize + best.col as isize - reach - 4,
        y as isize + best.row as isize - reach - 4,
    );
    let (ww, wh) = (w + 2 * reach as usize + 8, h + 2 * reach as usize + 8);
    let s = if range > 0
        && (wx0 < 0 || wy0 < 0 || wx0 as usize + ww > ref_w || wy0 as usize + wh > ref_h)
    {
        if win.len() < ww * wh {
            win.resize(ww * wh, P::default());
        }
        emu_edge(win, ww, refp, ref_stride, ref_w, ref_h, wx0, wy0, ww, wh);
        Search {
            win: &win[..],
            win_x: wx0,
            win_y: wy0,
            win_w: ww,
            win_h: wh,
            ..s
        }
    } else {
        s
    };
    if range > 0 {
        let wr_lo = (best.row - range).max(row_lo);
        let wr_hi = (best.row + range).min(row_hi);
        let wc_lo = (best.col - range).max(col_lo);
        let wc_hi = (best.col + range).min(col_hi);
        let in_window = |dy: i32, dx: i32| dy >= wr_lo && dy <= wr_hi && dx >= wc_lo && dx <= wc_hi;
        let mut max_step = 1i32;
        while max_step * 2 <= range.min(params.max_ring) {
            max_step *= 2;
        }
        // Rings of radius 1, 2, 4, .., max_step around the start; the best
        // point of all rings wins, so a nearby minimum is not abandoned for
        // a lucky far sample.
        let (cy, cx) = (best.row, best.col);
        let mut found = 1;
        let mut ring = 1;
        while ring <= max_step {
            for &(oy, ox) in &SQUARE {
                let (dy, dx) = (cy + oy * ring, cx + ox * ring);
                if !in_window(dy, dx) {
                    continue;
                }
                if let Some((c, d)) = s.eval_full(edge, dy, dx, best.cost) {
                    best = Best {
                        row: dy,
                        col: dx,
                        cost: c,
                        dist: d,
                    };
                    found = ring;
                }
            }
            ring *= 2;
        }
        // Descending-step refinement from the best ring point.
        let mut step = (found / 2).max(1);
        while step >= 1 {
            for _ in 0..params.max_iters.max(1) {
                let (cy, cx) = (best.row, best.col);
                let mut moved = false;
                for &(oy, ox) in &SQUARE {
                    let (dy, dx) = (cy + oy * step, cx + ox * step);
                    if !in_window(dy, dx) {
                        continue;
                    }
                    if let Some((c, d)) = s.eval_full(edge, dy, dx, best.cost) {
                        best = Best {
                            row: dy,
                            col: dx,
                            cost: c,
                            dist: d,
                        };
                        moved = true;
                    }
                }
                if !moved {
                    break;
                }
            }
            step /= 2;
        }
    }

    let mut mv = Best {
        row: best.row * 8,
        col: best.col * 8,
        cost: best.cost,
        dist: best.dist,
    };

    // (3) Sub-sample refinement.
    if params.subpel > 0 {
        // Rows of an earlier search (another block or plane) do not apply.
        cols.clear();
        let (r8_lo, r8_hi) = (row_lo * 8, row_hi * 8);
        let (c8_lo, c8_hi) = (col_lo * 8, col_hi * 8);
        let src_blk = s.src_block();
        {
            let (blk, bs) = s.ref_block(edge, best.row, best.col);
            let d = satd(src_blk, src_stride, blk, bs, w, h);
            mv.dist = d;
            mv.cost = d.saturating_add(mv_cost(mv.row, mv.col));
        }
        let levels = usize::from(params.subpel.min(3));
        for &st in &[4i32, 2, 1][..levels] {
            let (cy, cx) = (mv.row, mv.col);
            // Costs of the four axis neighbours (SQUARE starts with them).
            let mut cross = [u32::MAX; 4];
            for (i, &(oy, ox)) in SQUARE.iter().enumerate() {
                if params.subpel_pruned && i >= 4 {
                    // Only the diagonal between the better vertical and
                    // the better horizontal neighbour.
                    let dy = if cross[0] <= cross[3] { -1 } else { 1 };
                    let dx = if cross[1] <= cross[2] { -1 } else { 1 };
                    if (oy, ox) != (dy, dx) {
                        continue;
                    }
                }
                let (r, c) = (cy + oy * st, cx + ox * st);
                if r < r8_lo || r > r8_hi || c < c8_lo || c > c8_hi {
                    continue;
                }
                let rate = mv_cost(r, c);
                if rate >= mv.cost {
                    continue;
                }
                s.predict_shared(cols, pred, edge, (r, c));
                let d = satd(src_blk, src_stride, &pred[..], w, w, h);
                let cost = d.saturating_add(rate);
                if i < 4 {
                    cross[i] = cost;
                }
                if cost < mv.cost {
                    mv = Best {
                        row: r,
                        col: c,
                        cost,
                        dist: d,
                    };
                }
            }
        }
    }

    MeResult {
        mv: (mv.row, mv.col),
        cost: mv.cost,
        dist: mv.dist,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MeParams, MeScratch, block_mean, block_sum, block_variance, diff_variance, downscale_2x,
        downscale_2x_dims, energy_i32, motion_search, sad, sad_limited, satd, sse, sse_i16,
        sse_i32,
    };
    use crate::dsp::Pixel;
    use crate::dsp::mc::{emu_edge, put};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    const SHAPES: [(usize, usize); 22] = [
        (4, 4),
        (4, 8),
        (8, 4),
        (8, 8),
        (8, 16),
        (16, 8),
        (16, 16),
        (16, 32),
        (32, 16),
        (32, 32),
        (32, 64),
        (64, 32),
        (64, 64),
        (64, 128),
        (128, 64),
        (128, 128),
        (4, 16),
        (16, 4),
        (8, 32),
        (32, 8),
        (16, 64),
        (64, 16),
    ];

    fn rand_plane<P: Pixel>(rng: &mut Rng, n: usize, bd: u32) -> Vec<P> {
        (0..n)
            .map(|_| P::from_i32(rng.below(1 << bd) as i32))
            .collect()
    }

    fn naive_sad<P: Pixel>(a: &[P], sa: usize, b: &[P], sb: usize, w: usize, h: usize) -> u64 {
        let mut s = 0u64;
        for r in 0..h {
            for c in 0..w {
                s += (a[r * sa + c].to_i32() - b[r * sb + c].to_i32()).unsigned_abs() as u64;
            }
        }
        s
    }

    fn naive_sse<P: Pixel>(a: &[P], sa: usize, b: &[P], sb: usize, w: usize, h: usize) -> u64 {
        let mut s = 0u64;
        for r in 0..h {
            for c in 0..w {
                let d = (a[r * sa + c].to_i32() - b[r * sb + c].to_i32()) as i64;
                s += (d * d) as u64;
            }
        }
        s
    }

    fn naive_satd<P: Pixel>(a: &[P], sa: usize, b: &[P], sb: usize, w: usize, h: usize) -> u64 {
        let n = if w.is_multiple_of(8) && h.is_multiple_of(8) {
            8
        } else {
            4
        };
        let hm = |i: usize, j: usize| {
            if (i & j).count_ones().is_multiple_of(2) {
                1
            } else {
                -1
            }
        };
        let mut total = 0u64;
        for by in (0..h).step_by(n) {
            for bx in (0..w).step_by(n) {
                let mut s = 0u64;
                for u in 0..n {
                    for v in 0..n {
                        let mut c = 0i64;
                        for i in 0..n {
                            for j in 0..n {
                                let d = a[(by + i) * sa + bx + j].to_i32()
                                    - b[(by + i) * sb + bx + j].to_i32();
                                c += (hm(u, i) * hm(v, j) * d) as i64;
                            }
                        }
                        s += c.unsigned_abs();
                    }
                }
                total += (s + 1) >> 1;
            }
        }
        total
    }

    fn kernels_for<P: Pixel>(bd: u32, seed: u64) {
        let mut rng = Rng(seed);
        for &(w, h) in &SHAPES {
            for extra in [0usize, 3] {
                let (sa, sb) = (w + extra, w + 2 * extra + 1);
                let a = rand_plane::<P>(&mut rng, sa * h, bd);
                let mut b = rand_plane::<P>(&mut rng, sb * h, bd);
                // Also a near-identical pair (small differences).
                if extra == 3 {
                    for r in 0..h {
                        for c in 0..w {
                            let v = a[r * sa + c].to_i32() + rng.below(5) as i32 - 2;
                            b[r * sb + c] = P::from_i32(v.clamp(0, (1 << bd) - 1));
                        }
                    }
                }
                assert_eq!(
                    sad(&a, sa, &b, sb, w, h) as u64,
                    naive_sad(&a, sa, &b, sb, w, h),
                    "sad {w}x{h}"
                );
                assert_eq!(sse(&a, sa, &b, sb, w, h), naive_sse(&a, sa, &b, sb, w, h));
                assert_eq!(
                    satd(&a, sa, &b, sb, w, h) as u64,
                    naive_satd(&a, sa, &b, sb, w, h),
                    "satd {w}x{h}"
                );
                let full = sad(&a, sa, &b, sb, w, h);
                let lim = sad_limited(&a, sa, &b, sb, w, h, full / 2);
                assert!(lim > full / 2 || full == 0);
                assert_eq!(sad_limited(&a, sa, &b, sb, w, h, full), full);

                // Mean / variance.
                let mut s = 0u64;
                let mut q = 0u64;
                let mut ds = 0i64;
                for r in 0..h {
                    for c in 0..w {
                        let v = a[r * sa + c].to_i32() as u64;
                        s += v;
                        q += v * v;
                        ds += (a[r * sa + c].to_i32() - b[r * sb + c].to_i32()) as i64;
                    }
                }
                let n = (w * h) as u64;
                assert_eq!(block_sum(&a, sa, w, h), (s as u32, q));
                assert_eq!(block_mean(&a, sa, w, h) as u64, (s + n / 2) / n);
                assert_eq!(block_variance(&a, sa, w, h), q - s * s / n);
                let e = naive_sse(&a, sa, &b, sb, w, h);
                let ds2 = ds.unsigned_abs() * ds.unsigned_abs();
                assert_eq!(diff_variance(&a, sa, &b, sb, w, h), (e, e - ds2 / n));
            }
        }
    }

    #[test]
    fn kernels_u8() {
        kernels_for::<u8>(8, 0x1234_5678_9abc_def1);
    }

    #[test]
    fn kernels_u16() {
        kernels_for::<u16>(10, 0x0fed_cba9_8765_4321);
        kernels_for::<u16>(12, 0x5555_aaaa_1234_9876);
    }

    #[test]
    fn kernels_extremes() {
        // Worst-case magnitudes do not overflow.
        let a = vec![4095u16; 128 * 128];
        let b = vec![0u16; 128 * 128];
        assert_eq!(sad(&a, 128, &b, 128, 128, 128), 4095 * 128 * 128);
        assert_eq!(sse(&a, 128, &b, 128, 128, 128), 4095 * 4095 * 128 * 128);
        assert_eq!(satd(&a, 128, &b, 128, 128, 128), 4095 * 128 * 128 / 2);
        assert_eq!(block_variance(&a, 128, 128, 128), 0);
    }

    #[test]
    fn residual_sse() {
        let mut rng = Rng(77);
        let a: Vec<i16> = (0..1000).map(|_| rng.next() as i16).collect();
        let b: Vec<i16> = (0..1000).map(|_| rng.next() as i16).collect();
        let n: u64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| ((x as i64 - y as i64) * (x as i64 - y as i64)) as u64)
            .sum();
        assert_eq!(sse_i16(&a, &b), n);
        let a: Vec<i32> = (0..300)
            .map(|_| (rng.next() % 200_001) as i32 - 100_000)
            .collect();
        let b: Vec<i32> = (0..300)
            .map(|_| (rng.next() % 200_001) as i32 - 100_000)
            .collect();
        let n: u64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| ((x as i64 - y as i64) * (x as i64 - y as i64)) as u64)
            .sum();
        assert_eq!(sse_i32(&a, &b), n);
        let e: u64 = a.iter().map(|&x| (x as i64 * x as i64) as u64).sum();
        assert_eq!(energy_i32(&a), e);
    }

    #[test]
    fn downscale() {
        let src: Vec<u8> = (0..35).map(|i| (i * 7 % 256) as u8).collect();
        let (w, h) = (7, 5);
        let (ow, oh) = downscale_2x_dims(w, h);
        assert_eq!((ow, oh), (4, 3));
        let mut dst = vec![0u8; ow * oh];
        downscale_2x(&mut dst, ow, &src, w, w, h);
        for r in 0..oh {
            for c in 0..ow {
                let at = |y: usize, x: usize| src[y.min(h - 1) * w + x.min(w - 1)] as i32;
                let s = at(2 * r, 2 * c)
                    + at(2 * r, 2 * c + 1)
                    + at(2 * r + 1, 2 * c)
                    + at(2 * r + 1, 2 * c + 1);
                assert_eq!(dst[r * ow + c] as i32, (s + 2) >> 2);
            }
        }
    }

    /// A smooth random texture (box-blurred noise plus a little fine noise).
    fn texture<P: Pixel>(seed: u64, w: usize, h: usize, bd: u32) -> Vec<P> {
        let mut rng = Rng(seed);
        let max = (1i64 << bd) - 1;
        let mut v: Vec<i64> = (0..w * h).map(|_| rng.below(1024) as i64).collect();
        let rad = 3i64;
        for _ in 0..2 {
            let mut t = vec![0i64; w * h];
            for y in 0..h {
                for x in 0..w {
                    let mut s = 0;
                    for k in -rad..=rad {
                        let xx = (x as i64 + k).clamp(0, w as i64 - 1) as usize;
                        s += v[y * w + xx];
                    }
                    t[y * w + x] = s;
                }
            }
            for y in 0..h {
                for x in 0..w {
                    let mut s = 0;
                    for k in -rad..=rad {
                        let yy = (y as i64 + k).clamp(0, h as i64 - 1) as usize;
                        s += t[yy * w + x];
                    }
                    v[y * w + x] = s / 49;
                }
            }
        }
        let (lo, hi) = (*v.iter().min().unwrap_or(&0), *v.iter().max().unwrap_or(&1));
        v.iter()
            .map(|&e| {
                let s = (e - lo) * max / (hi - lo).max(1) + rng.below(3) as i64 - 1;
                P::from_i32(s.clamp(0, max) as i32)
            })
            .collect()
    }

    /// Builds the source block that `mv` (1/8) predicts at `(x, y)` with
    /// the regular filter, through the same edge-clamped path as a decoder.
    fn predicted<P: Pixel>(
        refp: &[P],
        rw: usize,
        rh: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        mv: (i32, i32),
        bd: u32,
    ) -> Vec<P> {
        let py = y as isize * 8 + mv.0 as isize;
        let px = x as isize * 8 + mv.1 as isize;
        let es = w + 7;
        let mut edge = vec![P::default(); es * (h + 7)];
        emu_edge(
            &mut edge,
            es,
            refp,
            rw,
            rw,
            rh,
            (px >> 3) - 3,
            (py >> 3) - 3,
            es,
            h + 7,
        );
        let mut out = vec![P::default(); w * h];
        let fx = if w <= 4 { 4 } else { 0 };
        let fy = if h <= 4 { 4 } else { 0 };
        put(
            &mut out,
            w,
            &edge,
            3 * es + 3,
            es,
            w,
            h,
            ((px & 7) * 2) as usize,
            ((py & 7) * 2) as usize,
            fx,
            fy,
            bd,
        );
        out
    }

    /// Embeds `blk` into a plane-sized source at `(x, y)`.
    fn source<P: Pixel>(blk: &[P], rw: usize, rh: usize, x: usize, y: usize, w: usize) -> Vec<P> {
        let mut s = vec![P::default(); rw * rh];
        for (r, row) in blk.chunks(w).enumerate() {
            s[(y + r) * rw + x..][..w].copy_from_slice(row);
        }
        s
    }

    fn check_search<P: Pixel>(
        bd: u32,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        mv: (i32, i32),
        subpel: u8,
        cands: &[(i32, i32)],
    ) {
        let (rw, rh) = (192, 144);
        let refp = texture::<P>(0x9e37_79b9 ^ (w * 31 + h) as u64, rw, rh, bd);
        let blk = predicted(&refp, rw, rh, x, y, w, h, mv, bd);
        let src = source(&blk, rw, rh, x, y, w);
        let params = MeParams {
            subpel,
            ..MeParams::default()
        };
        let mut scratch = MeScratch::<P>::new();
        let r = motion_search(
            &mut scratch,
            &src,
            rw,
            x,
            y,
            w,
            h,
            &refp,
            rw,
            rw,
            rh,
            cands,
            &params,
            &|_, _| 0,
            bd,
        );
        // The true MV predicts the block exactly; the search must find a
        // vector at least as good (usually the same one).
        let got = predicted(&refp, rw, rh, x, y, w, h, r.mv, bd);
        assert_eq!(
            satd(&got, w, &blk, w, w, h),
            0,
            "{w}x{h} at ({x},{y}) want {mv:?} got {r:?}"
        );
        assert_eq!(r.dist, 0);
        if subpel == 3 || (mv.0 % 8 == 0 && mv.1 % 8 == 0) {
            // The texture is unique enough that only the true MV is exact,
            // unless the block lies wholly in replicated edge samples.
            let px = x as i32 + (mv.1 >> 3);
            let py = y as i32 + (mv.0 >> 3);
            if px > 0 && py > 0 && px + (w as i32) < rw as i32 && py + (h as i32) < rh as i32 {
                assert_eq!(r.mv, mv);
            }
        }
    }

    #[test]
    fn finds_fullpel_translation() {
        check_search::<u8>(8, 64, 48, 16, 16, (-5 * 8, 7 * 8), 0, &[]);
        check_search::<u8>(8, 80, 40, 32, 16, (11 * 8, -9 * 8), 2, &[]);
        check_search::<u8>(8, 64, 64, 8, 8, (3 * 8, 2 * 8), 3, &[]);
        check_search::<u16>(10, 64, 48, 16, 16, (-6 * 8, -4 * 8), 1, &[]);
        check_search::<u16>(12, 60, 40, 64, 64, (9 * 8, 12 * 8), 0, &[]);
        // A far vector found from a nearby candidate.
        check_search::<u8>(8, 32, 32, 16, 16, (40 * 8, 90 * 8), 0, &[(38 * 8, 93 * 8)]);
    }

    #[test]
    fn finds_subpel_translation() {
        check_search::<u8>(8, 64, 48, 16, 16, (13, -19), 3, &[]);
        check_search::<u8>(8, 64, 48, 4, 16, (-21, 6), 3, &[]);
        check_search::<u8>(8, 64, 48, 16, 4, (5, 3), 3, &[]);
        check_search::<u8>(8, 64, 48, 32, 32, (36, -50), 3, &[(32, -48)]);
        check_search::<u16>(10, 64, 48, 16, 16, (-13, 27), 3, &[]);
        check_search::<u16>(12, 64, 48, 8, 8, (7, -1), 3, &[]);
    }

    #[test]
    fn finds_vectors_across_frame_edges() {
        // Partly outside the top-left corner.
        check_search::<u8>(8, 8, 4, 16, 16, (-10 * 8, -12 * 8), 0, &[]);
        check_search::<u8>(8, 8, 4, 16, 16, (-75, -101), 3, &[(-80, -96)]);
        // Bottom-right, block at the frame edge pointing out.
        check_search::<u8>(8, 176, 128, 16, 16, (5 * 8 + 3, 6 * 8 - 3), 3, &[]);
        check_search::<u16>(10, 160, 112, 32, 32, (13 * 8, 21 * 8), 2, &[]);
        check_search::<u16>(10, 0, 0, 8, 8, (-3, -5), 3, &[]);
    }

    #[test]
    fn respects_window_and_rate() {
        let (rw, rh) = (64usize, 48usize);
        let refp = texture::<u8>(5, rw, rh, 8);
        let src = refp.clone();
        let mut scratch = MeScratch::<u8>::new();
        let params = MeParams {
            border: 8,
            subpel: 3,
            ..MeParams::default()
        };
        // Far-out candidates are clamped into the border window.
        let r = motion_search(
            &mut scratch,
            &src,
            rw,
            0,
            0,
            16,
            16,
            &refp,
            rw,
            rw,
            rh,
            &[(-8000, -8000), (9000, 9000)],
            &params,
            &|_, _| 0,
            8,
        );
        assert_eq!(r.mv, (0, 0));
        assert_eq!(r.dist, 0);
        // A rate term that penalises every non-zero MV keeps the zero MV even
        // with an identical-looking flat region.
        let flat = vec![100u8; rw * rh];
        let r = motion_search(
            &mut scratch,
            &flat,
            rw,
            16,
            16,
            16,
            16,
            &flat,
            rw,
            rw,
            rh,
            &[(24, 16)],
            &params,
            &|r, c| r.unsigned_abs() + c.unsigned_abs(),
            8,
        );
        assert_eq!((r.mv, r.cost, r.dist), ((0, 0), 0, 0));
        // Every result stays within the window.
        for &(cy, cx) in &[(-400, -400), (400, 400)] {
            let r = motion_search(
                &mut scratch,
                &src,
                rw,
                48,
                32,
                16,
                16,
                &flat,
                rw,
                rw,
                rh,
                &[(cy, cx)],
                &params,
                &|_, _| 0,
                8,
            );
            let (py, px) = (32 * 8 + r.mv.0, 48 * 8 + r.mv.1);
            assert!(px >= -(16 + 8) * 8 && px <= (rw as i32 + 8) * 8, "{r:?}");
            assert!(py >= -(16 + 8) * 8 && py <= (rh as i32 + 8) * 8, "{r:?}");
        }
    }
}
