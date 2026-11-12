//! NEON luma and chroma prediction of whole blocks, one routine for each
//! position, as FFmpeg's `h264qpel_neon.S` and `h264cmc_neon.S` have one
//! for each: the routine for a block's offsets is picked from a table, and
//! it makes each row of the block in registers, both sources of a quarter
//! sample included, and stores it once.
//!
//! # Memory
//!
//! These routines load and store through pointers. [`luma_block`],
//! [`chroma_block`], the pair functions and [`chroma_planes`] check,
//! before any load, that every byte the chosen routine loads or stores
//! lies inside the slices they were given, and return `false` when one
//! does not. The routines derive every pointer from those slices within
//! the checked span. The plain twins in [`crate::scalar`] are the
//! reference, and the tests in [`crate::mc`] compare the two at every
//! quarter and eighth offset.
#![allow(unsafe_code)]

use core::arch::aarch64::{
    int16x8_t, uint8x8_t, uint8x16_t, vcombine_u8, vdup_n_u8, vdupq_n_s16, vdupq_n_u8, vget_low_u8,
    vld1_u8, vld1q_u8, vmlal_u8, vmull_u8, vqrshrun_high_n_s16, vqrshrun_n_s16, vrhadd_u8,
    vrhaddq_u8, vrshrn_n_u16,
};

use crate::mc_block::{Block, ChromaSource};
use crate::neon_qpel::{
    centre8, load4, put8, put16, row_taps8, row_taps16, store4, taps8, taps8_high,
};

/// Full samples.
const FULL: u8 = 0;
/// Horizontal half samples.
const HALF_H: u8 = 1;
/// Vertical half samples.
const HALF_V: u8 = 2;
/// Centre half samples.
const CENTRE: u8 = 3;
/// No second source: the position is a full or half sample.
const NONE: u8 = 4;

/// One source of a block's rows, made a row at a time: where its next row
/// is read from, and the rows a column filter holds.
#[derive(Clone, Copy)]
struct Feed<V> {
    p: *const u8,
    rows: [V; 6],
}

/// The rows of a 16-wide source: bytes for the vertical filter, or the
/// unrounded horizontal sums, low and high halves, for the centre one.
#[derive(Clone, Copy)]
struct Wide {
    bytes: uint8x16_t,
    low: int16x8_t,
    high: int16x8_t,
}

/// Horizontal half samples of sixteen samples from the 32 bytes at `p`.
///
/// # Safety
///
/// 32 bytes from `p` must be readable.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn half_h16(p: *const u8) -> uint8x16_t {
    // SAFETY: the caller's promise.
    let t = unsafe { row_taps16(p) };
    let low = taps8([
        vget_low_u8(t[0]),
        vget_low_u8(t[1]),
        vget_low_u8(t[2]),
        vget_low_u8(t[3]),
        vget_low_u8(t[4]),
        vget_low_u8(t[5]),
    ]);
    vqrshrun_high_n_s16::<5>(vqrshrun_n_s16::<5>(low), taps8_high(t))
}

/// The unrounded horizontal sums of sixteen samples from `p`, as for
/// [`half_h16`].
///
/// # Safety
///
/// As for [`half_h16`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn sums16(p: *const u8) -> (int16x8_t, int16x8_t) {
    // SAFETY: the caller's promise.
    let t = unsafe { row_taps16(p) };
    let low = taps8([
        vget_low_u8(t[0]),
        vget_low_u8(t[1]),
        vget_low_u8(t[2]),
        vget_low_u8(t[3]),
        vget_low_u8(t[4]),
        vget_low_u8(t[5]),
    ]);
    (low, taps8_high(t))
}

/// A feed of kind `KIND` whose samples start `X` right of and `Y` below
/// `src`, the block's top-left full sample. A column filter has read the
/// five rows before its first output row's last tap.
///
/// # Safety
///
/// Every row the feed reads for an `h`-row block must be readable: see
/// [`luma_reach`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn start16<const KIND: u8, const X: usize, const Y: usize>(
    src: *const u8,
    stride: usize,
) -> Feed<Wide> {
    let blank = Wide {
        bytes: vdupq_n_u8(0),
        low: vdupq_n_s16(0),
        high: vdupq_n_s16(0),
    };
    let mut feed = Feed {
        p: src,
        rows: [blank; 6],
    };
    // SAFETY: the offsets stay inside the rows the caller promises.
    unsafe {
        let base = src.add(Y * stride + X);
        match KIND {
            HALF_V => {
                let mut p = base.sub(2 * stride);
                for slot in 1..6 {
                    feed.rows[slot].bytes = vld1q_u8(p);
                    p = p.add(stride);
                }
                feed.p = p;
            }
            CENTRE => {
                let mut p = base.sub(2 * stride + 2);
                for slot in 1..6 {
                    (feed.rows[slot].low, feed.rows[slot].high) = sums16(p);
                    p = p.add(stride);
                }
                feed.p = p;
            }
            HALF_H => feed.p = base.sub(2),
            _ => feed.p = base,
        }
    }
    feed
}

/// The feed's next row of sixteen samples.
///
/// # Safety
///
/// As for [`start16`], for one row more than the feed has made.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn next16<const KIND: u8>(feed: &mut Feed<Wide>, stride: usize) -> uint8x16_t {
    // SAFETY: the caller's promise.
    unsafe {
        let p = feed.p;
        feed.p = p.add(stride);
        match KIND {
            HALF_H => half_h16(p),
            HALF_V => {
                let r = &mut feed.rows;
                *r = [r[1], r[2], r[3], r[4], r[5], r[5]];
                r[5].bytes = vld1q_u8(p);
                let low = taps8([
                    vget_low_u8(r[0].bytes),
                    vget_low_u8(r[1].bytes),
                    vget_low_u8(r[2].bytes),
                    vget_low_u8(r[3].bytes),
                    vget_low_u8(r[4].bytes),
                    vget_low_u8(r[5].bytes),
                ]);
                let high = taps8_high([
                    r[0].bytes, r[1].bytes, r[2].bytes, r[3].bytes, r[4].bytes, r[5].bytes,
                ]);
                vqrshrun_high_n_s16::<5>(vqrshrun_n_s16::<5>(low), high)
            }
            CENTRE => {
                let r = &mut feed.rows;
                *r = [r[1], r[2], r[3], r[4], r[5], r[5]];
                (r[5].low, r[5].high) = sums16(p);
                vcombine_u8(
                    centre8([r[0].low, r[1].low, r[2].low, r[3].low, r[4].low, r[5].low]),
                    centre8([
                        r[0].high, r[1].high, r[2].high, r[3].high, r[4].high, r[5].high,
                    ]),
                )
            }
            _ => vld1q_u8(p),
        }
    }
}

/// [`start16`] for a block 8 wide: the vertical filter's rows are bytes in
/// the low half of each vector, the centre one's sums in the low lanes.
///
/// # Safety
///
/// As for [`start16`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn start8<const KIND: u8, const X: usize, const Y: usize>(
    src: *const u8,
    stride: usize,
) -> Feed<(uint8x8_t, int16x8_t)> {
    let mut feed = Feed {
        p: src,
        rows: [(vdup_n_u8(0), vdupq_n_s16(0)); 6],
    };
    // SAFETY: as in `start16`.
    unsafe {
        let base = src.add(Y * stride + X);
        match KIND {
            HALF_V => {
                let mut p = base.sub(2 * stride);
                for slot in 1..6 {
                    feed.rows[slot].0 = vld1_u8(p);
                    p = p.add(stride);
                }
                feed.p = p;
            }
            CENTRE => {
                let mut p = base.sub(2 * stride + 2);
                for slot in 1..6 {
                    feed.rows[slot].1 = taps8(row_taps8(p));
                    p = p.add(stride);
                }
                feed.p = p;
            }
            HALF_H => feed.p = base.sub(2),
            _ => feed.p = base,
        }
    }
    feed
}

/// [`next16`] for eight samples.
///
/// # Safety
///
/// As for [`next16`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn next8<const KIND: u8>(
    feed: &mut Feed<(uint8x8_t, int16x8_t)>,
    stride: usize,
) -> uint8x8_t {
    // SAFETY: the caller's promise.
    unsafe {
        let p = feed.p;
        feed.p = p.add(stride);
        match KIND {
            HALF_H => vqrshrun_n_s16::<5>(taps8(row_taps8(p))),
            HALF_V => {
                let r = &mut feed.rows;
                *r = [r[1], r[2], r[3], r[4], r[5], r[5]];
                r[5].0 = vld1_u8(p);
                vqrshrun_n_s16::<5>(taps8([r[0].0, r[1].0, r[2].0, r[3].0, r[4].0, r[5].0]))
            }
            CENTRE => {
                let r = &mut feed.rows;
                *r = [r[1], r[2], r[3], r[4], r[5], r[5]];
                r[5].1 = taps8(row_taps8(p));
                centre8([r[0].1, r[1].1, r[2].1, r[3].1, r[4].1, r[5].1])
            }
            _ => vld1_u8(p),
        }
    }
}

/// The routine for one luma position: `dst` and its row distance, the
/// block's top-left full sample and its row distance, and the height.
type LumaFn = unsafe fn(*mut u8, usize, *const u8, usize, usize);

/// One luma position of a 16-wide block: the source of kind `K1` at
/// (`X1`, `Y1`), averaged with the one of kind `K2` at (`X2`, `Y2`) unless
/// `K2` is [`NONE`], written to `dst` or, with `AVG`, averaged into it.
///
/// # Safety
///
/// `h` rows of 16 from `dst` must be writable, and every row the sources
/// read must be readable: see [`luma_reach`].
#[target_feature(enable = "neon")]
unsafe fn luma16<
    const AVG: bool,
    const K1: u8,
    const X1: usize,
    const Y1: usize,
    const K2: u8,
    const X2: usize,
    const Y2: usize,
>(
    dst: *mut u8,
    dst_stride: usize,
    src: *const u8,
    stride: usize,
    h: usize,
) {
    // SAFETY: the caller's promise.
    unsafe {
        let mut one = start16::<K1, X1, Y1>(src, stride);
        let mut two = start16::<K2, X2, Y2>(src, stride);
        let mut out = dst;
        for _ in 0..h {
            let mut row = next16::<K1>(&mut one, stride);
            if K2 != NONE {
                row = vrhaddq_u8(row, next16::<K2>(&mut two, stride));
            }
            put16::<AVG>(out, row);
            out = out.add(dst_stride);
        }
    }
}

/// [`luma16`] for a block 8 wide.
///
/// # Safety
///
/// As for [`luma16`], with rows of 8.
#[target_feature(enable = "neon")]
unsafe fn luma8<
    const AVG: bool,
    const K1: u8,
    const X1: usize,
    const Y1: usize,
    const K2: u8,
    const X2: usize,
    const Y2: usize,
>(
    dst: *mut u8,
    dst_stride: usize,
    src: *const u8,
    stride: usize,
    h: usize,
) {
    // SAFETY: the caller's promise.
    unsafe {
        let mut one = start8::<K1, X1, Y1>(src, stride);
        let mut two = start8::<K2, X2, Y2>(src, stride);
        let mut out = dst;
        for _ in 0..h {
            let mut row = next8::<K1>(&mut one, stride);
            if K2 != NONE {
                row = vrhadd_u8(row, next8::<K2>(&mut two, stride));
            }
            put8::<AVG>(out, row);
            out = out.add(dst_stride);
        }
    }
}

/// The sixteen luma routines of one width and kind of write, indexed by
/// `yf * 4 + xf`. Each quarter position's two sources are those of clause
/// 8.4.2.2.1: see [`crate::mc_block::quarter_sources`].
macro_rules! luma_table {
    ($f:ident, $avg:expr) => {
        [
            $f::<$avg, FULL, 0, 0, NONE, 0, 0>,
            $f::<$avg, FULL, 0, 0, HALF_H, 0, 0>,
            $f::<$avg, HALF_H, 0, 0, NONE, 0, 0>,
            $f::<$avg, HALF_H, 0, 0, FULL, 1, 0>,
            $f::<$avg, FULL, 0, 0, HALF_V, 0, 0>,
            $f::<$avg, HALF_H, 0, 0, HALF_V, 0, 0>,
            $f::<$avg, HALF_H, 0, 0, CENTRE, 0, 0>,
            $f::<$avg, HALF_H, 0, 0, HALF_V, 1, 0>,
            $f::<$avg, HALF_V, 0, 0, NONE, 0, 0>,
            $f::<$avg, HALF_V, 0, 0, CENTRE, 0, 0>,
            $f::<$avg, CENTRE, 0, 0, NONE, 0, 0>,
            $f::<$avg, CENTRE, 0, 0, HALF_V, 1, 0>,
            $f::<$avg, HALF_V, 0, 0, FULL, 0, 1>,
            $f::<$avg, HALF_V, 0, 0, HALF_H, 0, 1>,
            $f::<$avg, CENTRE, 0, 0, HALF_H, 0, 1>,
            $f::<$avg, HALF_V, 1, 0, HALF_H, 0, 1>,
        ]
    };
}

static LUMA16: [LumaFn; 16] = luma_table!(luma16, false);
static LUMA16_AVG: [LumaFn; 16] = luma_table!(luma16, true);
static LUMA8: [LumaFn; 16] = luma_table!(luma8, false);
static LUMA8_AVG: [LumaFn; 16] = luma_table!(luma8, true);

/// Where a luma block's routine reads, when that and the block's rows in
/// `dst` all fit: the block's top-left sample. Each direction is read
/// beyond the block only when it has an offset. Without a horizontal one,
/// every source reads the `w` samples of the block's columns; with one,
/// 32 bytes (16 wide) or 16 (8 wide) from two samples left of each row,
/// which covers each source's reach. Without a vertical one, the sources
/// read the block's rows; with one, two rows above them and three below.
#[inline]
fn luma_reach(block: &Block<'_>, frac: (u8, u8), dst: (&[u8], usize)) -> Option<*const u8> {
    let Block {
        src,
        stride,
        at: (x, y),
        size: (w, h),
    } = *block;
    let (dst, dst_stride) = dst;
    if (w != 8 && w != 16) || h == 0 || h > 16 || frac.0 > 3 || frac.1 > 3 {
        return None;
    }
    if dst_stride < w || (h - 1) * dst_stride + w > dst.len() {
        return None;
    }
    // Samples read left of the block and in each row from there.
    let (left, span) = if frac.0 == 0 { (0, w) } else { (2, 2 * w) };
    // Rows read above the block's first and below its last.
    let (above, below) = if frac.1 == 0 { (0, 0) } else { (2, 3) };
    if x < left || y < above || (frac.0 != 0 && stride < w + 5) {
        return None;
    }
    let last = (y + h - 1 + below)
        .checked_mul(stride)?
        .checked_add(x - left + span)?;
    if last > src.len() {
        return None;
    }
    // In bounds: `y * stride + x` is below `last`.
    Some(src[y * stride + x..].as_ptr())
}

/// The table entry for a block `w` wide at `frac`.
#[inline]
fn luma_fn(w: usize, avg: bool, frac: (u8, u8)) -> LumaFn {
    let index = usize::from(frac.1 & 3) * 4 + usize::from(frac.0 & 3);
    match (w == 16, avg) {
        (true, false) => LUMA16[index],
        (true, true) => LUMA16_AVG[index],
        (false, false) => LUMA8[index],
        (false, true) => LUMA8_AVG[index],
    }
}

/// See [`crate::mc::luma_block`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn luma_block(
    dst: &mut [u8],
    dst_stride: usize,
    block: Block<'_>,
    frac: (u8, u8),
) -> bool {
    let Some(src) = luma_reach(&block, frac, (dst, dst_stride)) else {
        return false;
    };
    let (w, h) = block.size;
    // SAFETY: `luma_reach` checked every byte the routine reads and writes.
    unsafe { luma_fn(w, false, frac)(dst.as_mut_ptr(), dst_stride, src, block.stride, h) };
    true
}

/// See [`crate::mc::luma_pair`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn luma_pair(
    dst: &mut [u8],
    dst_stride: usize,
    first: (Block<'_>, (u8, u8)),
    second: Option<(Block<'_>, (u8, u8))>,
) -> bool {
    let Some(one) = luma_reach(&first.0, first.1, (dst, dst_stride)) else {
        return false;
    };
    let two = match second {
        Some((block, frac)) if block.size == first.0.size => {
            match luma_reach(&block, frac, (dst, dst_stride)) {
                Some(src) => Some((src, block.stride, frac)),
                None => return false,
            }
        }
        Some(_) => return false,
        None => None,
    };
    let (w, h) = first.0.size;
    let out = dst.as_mut_ptr();
    // SAFETY: `luma_reach` checked every byte each routine reads and writes.
    unsafe {
        luma_fn(w, false, first.1)(out, dst_stride, one, first.0.stride, h);
        if let Some((src, stride, frac)) = two {
            luma_fn(w, true, frac)(out, dst_stride, src, stride, h);
        }
    }
    true
}

/// The routine for one chroma kind: `dst` and its row distance, the
/// block's top-left sample and its row distance, the height, and the
/// eighth offsets.
type ChromaFn = unsafe fn(*mut u8, usize, *const u8, usize, usize, (u8, u8));

/// Writes `row`, `W` samples wide, to `out`, or its rounded average with
/// what `out` holds.
///
/// # Safety
///
/// `W` bytes from `out` must be readable and writable.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn put_chroma<const W: usize, const AVG: bool>(out: *mut u8, row: uint8x8_t) {
    // SAFETY: the caller's promise.
    unsafe {
        if W == 8 {
            put8::<AVG>(out, row);
        } else {
            let row = if AVG { vrhadd_u8(load4(out), row) } else { row };
            store4(out, row);
        }
    }
}

/// A chroma block `W` wide, clause 8.4.2.2.2, two rows a step. `MODE` says
/// which offsets are not zero: bit 0 the horizontal, bit 1 the vertical.
/// With one of them zero its taps drop out: `(8 − f)·8` and `f·8` weigh
/// the two samples left, with the same rounding shift of 6.
///
/// # Safety
///
/// `h` is even; `h` rows of `W` from `dst` must be writable, and `h + 1`
/// rows of 9 bytes from `src` readable.
#[target_feature(enable = "neon")]
unsafe fn chroma<const W: usize, const AVG: bool, const MODE: u8>(
    dst: *mut u8,
    dst_stride: usize,
    src: *const u8,
    stride: usize,
    h: usize,
    (xf, yf): (u8, u8),
) {
    // SAFETY: the caller's promise.
    unsafe {
        let (mut p, mut out) = (src, dst);
        match MODE {
            3 => {
                let wa = vdup_n_u8((8 - xf) * (8 - yf));
                let wb = vdup_n_u8(xf * (8 - yf));
                let wc = vdup_n_u8((8 - xf) * yf);
                let wd = vdup_n_u8(xf * yf);
                let mut top = (vld1_u8(p), vld1_u8(p.add(1)));
                for _ in 0..h / 2 {
                    let mid = (vld1_u8(p.add(stride)), vld1_u8(p.add(stride + 1)));
                    let below = (vld1_u8(p.add(2 * stride)), vld1_u8(p.add(2 * stride + 1)));
                    let first = vmlal_u8(
                        vmlal_u8(vmlal_u8(vmull_u8(top.0, wa), top.1, wb), mid.0, wc),
                        mid.1,
                        wd,
                    );
                    let second = vmlal_u8(
                        vmlal_u8(vmlal_u8(vmull_u8(mid.0, wa), mid.1, wb), below.0, wc),
                        below.1,
                        wd,
                    );
                    put_chroma::<W, AVG>(out, vrshrn_n_u16::<6>(first));
                    put_chroma::<W, AVG>(out.add(dst_stride), vrshrn_n_u16::<6>(second));
                    top = below;
                    p = p.add(2 * stride);
                    out = out.add(2 * dst_stride);
                }
            }
            1 => {
                let wa = vdup_n_u8((8 - xf) * 8);
                let wb = vdup_n_u8(xf * 8);
                for _ in 0..h / 2 {
                    let first = vmlal_u8(vmull_u8(vld1_u8(p), wa), vld1_u8(p.add(1)), wb);
                    let below = p.add(stride);
                    let second = vmlal_u8(vmull_u8(vld1_u8(below), wa), vld1_u8(below.add(1)), wb);
                    put_chroma::<W, AVG>(out, vrshrn_n_u16::<6>(first));
                    put_chroma::<W, AVG>(out.add(dst_stride), vrshrn_n_u16::<6>(second));
                    p = p.add(2 * stride);
                    out = out.add(2 * dst_stride);
                }
            }
            2 => {
                let wa = vdup_n_u8((8 - yf) * 8);
                let wc = vdup_n_u8(yf * 8);
                let mut top = vld1_u8(p);
                for _ in 0..h / 2 {
                    let mid = vld1_u8(p.add(stride));
                    let below = vld1_u8(p.add(2 * stride));
                    let first = vmlal_u8(vmull_u8(top, wa), mid, wc);
                    let second = vmlal_u8(vmull_u8(mid, wa), below, wc);
                    put_chroma::<W, AVG>(out, vrshrn_n_u16::<6>(first));
                    put_chroma::<W, AVG>(out.add(dst_stride), vrshrn_n_u16::<6>(second));
                    top = below;
                    p = p.add(2 * stride);
                    out = out.add(2 * dst_stride);
                }
            }
            _ => {
                for _ in 0..h / 2 {
                    put_chroma::<W, AVG>(out, vld1_u8(p));
                    put_chroma::<W, AVG>(out.add(dst_stride), vld1_u8(p.add(stride)));
                    p = p.add(2 * stride);
                    out = out.add(2 * dst_stride);
                }
            }
        }
    }
}

/// The chroma routines, indexed by width (4 then 8), then the kind of
/// write, then `MODE`.
static CHROMA: [[[ChromaFn; 4]; 2]; 2] = [
    [
        [
            chroma::<4, false, 0>,
            chroma::<4, false, 1>,
            chroma::<4, false, 2>,
            chroma::<4, false, 3>,
        ],
        [
            chroma::<4, true, 0>,
            chroma::<4, true, 1>,
            chroma::<4, true, 2>,
            chroma::<4, true, 3>,
        ],
    ],
    [
        [
            chroma::<8, false, 0>,
            chroma::<8, false, 1>,
            chroma::<8, false, 2>,
            chroma::<8, false, 3>,
        ],
        [
            chroma::<8, true, 0>,
            chroma::<8, true, 1>,
            chroma::<8, true, 2>,
            chroma::<8, true, 3>,
        ],
    ],
];

/// Where a chroma block's routine reads, when that and the block's rows in
/// `dst` all fit: the block's top-left sample. Each row reads 8 bytes from
/// its start and 8 from one later, and the row below it; the height is
/// even, as two rows are made a step.
#[inline]
fn chroma_reach(block: &Block<'_>, frac: (u8, u8), dst: (&[u8], usize)) -> Option<*const u8> {
    let Block {
        src,
        stride,
        at: (x, y),
        size: (w, h),
    } = *block;
    let (dst, dst_stride) = dst;
    if (w != 4 && w != 8) || h == 0 || h % 2 != 0 || frac.0 > 7 || frac.1 > 7 {
        return None;
    }
    if dst_stride < w || (h - 1) * dst_stride + w > dst.len() {
        return None;
    }
    let last = (y + h).checked_mul(stride)?.checked_add(x + 9)?;
    if last > src.len() {
        return None;
    }
    Some(src[y * stride + x..].as_ptr())
}

/// The routine for a block `w` wide at `frac`.
#[inline]
fn chroma_fn(w: usize, avg: bool, frac: (u8, u8)) -> ChromaFn {
    let mode = usize::from(frac.0 != 0) | usize::from(frac.1 != 0) << 1;
    CHROMA[usize::from(w == 8)][usize::from(avg)][mode]
}

/// See [`crate::mc::chroma_block`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn chroma_block(
    dst: &mut [u8],
    dst_stride: usize,
    block: Block<'_>,
    frac: (u8, u8),
) -> bool {
    let Some(src) = chroma_reach(&block, frac, (dst, dst_stride)) else {
        return false;
    };
    let (w, h) = block.size;
    // SAFETY: `chroma_reach` checked every byte the routine reads and writes.
    unsafe { chroma_fn(w, false, frac)(dst.as_mut_ptr(), dst_stride, src, block.stride, h, frac) };
    true
}

/// See [`crate::mc::chroma_pair`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn chroma_pair(
    dst: &mut [u8],
    dst_stride: usize,
    first: (Block<'_>, (u8, u8)),
    second: Option<(Block<'_>, (u8, u8))>,
) -> bool {
    let Some(one) = chroma_reach(&first.0, first.1, (dst, dst_stride)) else {
        return false;
    };
    let two = match second {
        Some((block, frac)) if block.size == first.0.size => {
            match chroma_reach(&block, frac, (dst, dst_stride)) {
                Some(src) => Some((src, block.stride, frac)),
                None => return false,
            }
        }
        Some(_) => return false,
        None => None,
    };
    let (w, h) = first.0.size;
    let out = dst.as_mut_ptr();
    // SAFETY: `chroma_reach` checked every byte each routine reads and writes.
    unsafe {
        chroma_fn(w, false, first.1)(out, dst_stride, one, first.0.stride, h, first.1);
        if let Some((src, stride, frac)) = two {
            chroma_fn(w, true, frac)(out, dst_stride, src, stride, h, frac);
        }
    }
    true
}

/// The rows of one chroma source, two a step, at eighth offsets whose
/// zero parts `MODE` names as [`chroma`]'s does; `MODE` [`NONE`] makes
/// nothing. `top` holds the row a step starts from.
struct ChromaRows<const MODE: u8> {
    p: *const u8,
    stride: usize,
    taps: [uint8x8_t; 4],
    top: (uint8x8_t, uint8x8_t),
}

impl<const MODE: u8> ChromaRows<MODE> {
    /// The source whose first row starts at `p`.
    ///
    /// # Safety
    ///
    /// As for [`chroma`]: the rows a block reads from `p` are readable.
    #[inline(always)]
    unsafe fn start(p: *const u8, stride: usize, (xf, yf): (u8, u8)) -> Self {
        // SAFETY: the caller's promise.
        unsafe {
            let mut rows = Self {
                p,
                stride,
                taps: [vdup_n_u8(0); 4],
                top: (vdup_n_u8(0), vdup_n_u8(0)),
            };
            match MODE {
                3 => {
                    rows.taps = [
                        vdup_n_u8((8 - xf) * (8 - yf)),
                        vdup_n_u8(xf * (8 - yf)),
                        vdup_n_u8((8 - xf) * yf),
                        vdup_n_u8(xf * yf),
                    ];
                    rows.top = (vld1_u8(p), vld1_u8(p.add(1)));
                }
                1 => rows.taps[..2].copy_from_slice(&[vdup_n_u8((8 - xf) * 8), vdup_n_u8(xf * 8)]),
                2 => {
                    rows.taps[..2].copy_from_slice(&[vdup_n_u8((8 - yf) * 8), vdup_n_u8(yf * 8)]);
                    rows.top.0 = vld1_u8(p);
                }
                _ => {}
            }
            rows
        }
    }

    /// The next two rows.
    ///
    /// # Safety
    ///
    /// The two rows, and for a vertical offset the one below them, are
    /// readable from where this source has got to.
    #[inline(always)]
    unsafe fn next2(&mut self) -> (uint8x8_t, uint8x8_t) {
        // SAFETY: the caller's promise.
        unsafe {
            let (p, stride) = (self.p, self.stride);
            self.p = p.add(2 * stride);
            let [wa, wb, wc, wd] = self.taps;
            match MODE {
                3 => {
                    let mid = (vld1_u8(p.add(stride)), vld1_u8(p.add(stride + 1)));
                    let below = (vld1_u8(p.add(2 * stride)), vld1_u8(p.add(2 * stride + 1)));
                    let top = self.top;
                    self.top = below;
                    let first = vmlal_u8(
                        vmlal_u8(vmlal_u8(vmull_u8(top.0, wa), top.1, wb), mid.0, wc),
                        mid.1,
                        wd,
                    );
                    let second = vmlal_u8(
                        vmlal_u8(vmlal_u8(vmull_u8(mid.0, wa), mid.1, wb), below.0, wc),
                        below.1,
                        wd,
                    );
                    (vrshrn_n_u16::<6>(first), vrshrn_n_u16::<6>(second))
                }
                1 => {
                    let below = p.add(stride);
                    let first = vmlal_u8(vmull_u8(vld1_u8(p), wa), vld1_u8(p.add(1)), wb);
                    let second = vmlal_u8(vmull_u8(vld1_u8(below), wa), vld1_u8(below.add(1)), wb);
                    (vrshrn_n_u16::<6>(first), vrshrn_n_u16::<6>(second))
                }
                2 => {
                    let mid = vld1_u8(p.add(stride));
                    let below = vld1_u8(p.add(2 * stride));
                    let top = self.top.0;
                    self.top.0 = below;
                    let first = vmlal_u8(vmull_u8(top, wa), mid, wb);
                    let second = vmlal_u8(vmull_u8(mid, wa), below, wb);
                    (vrshrn_n_u16::<6>(first), vrshrn_n_u16::<6>(second))
                }
                0 => (vld1_u8(p), vld1_u8(p.add(stride))),
                _ => (vdup_n_u8(0), vdup_n_u8(0)),
            }
        }
    }
}

/// Both planes' sources of one prediction for [`chroma_both`]: the Cb and
/// Cr samples the block starts from, their row distance, and the eighth
/// offsets.
#[derive(Clone, Copy)]
struct PlaneSources {
    cb: *const u8,
    cr: *const u8,
    stride: usize,
    frac: (u8, u8),
}

/// A chroma block `W` wide in both planes, from sources `one` at `A`'s
/// kind of offsets, rounded together with sources `two` at `B`'s when `B`
/// is not [`NONE`]. Each row is made in registers and stored once, Cb's
/// and Cr's in the same step from the same taps.
///
/// # Safety
///
/// `h` is even; `h` rows of `W` from each of `dst` must be writable, and
/// `h + 1` rows of 9 bytes from each source readable.
#[target_feature(enable = "neon")]
unsafe fn chroma_both<const W: usize, const A: u8, const B: u8>(
    dst: (*mut u8, *mut u8),
    dst_stride: usize,
    one: &PlaneSources,
    two: &PlaneSources,
    h: usize,
) {
    // SAFETY: the caller's promise.
    unsafe {
        let mut first_cb = ChromaRows::<A>::start(one.cb, one.stride, one.frac);
        let mut first_cr = ChromaRows::<A>::start(one.cr, one.stride, one.frac);
        let mut second_cb = ChromaRows::<B>::start(two.cb, two.stride, two.frac);
        let mut second_cr = ChromaRows::<B>::start(two.cr, two.stride, two.frac);
        let (mut cb, mut cr) = dst;
        for _ in 0..h / 2 {
            let (mut cb_upper, mut cb_lower) = first_cb.next2();
            let (mut cr_upper, mut cr_lower) = first_cr.next2();
            if B != NONE {
                let (upper, lower) = second_cb.next2();
                cb_upper = vrhadd_u8(cb_upper, upper);
                cb_lower = vrhadd_u8(cb_lower, lower);
                let (upper, lower) = second_cr.next2();
                cr_upper = vrhadd_u8(cr_upper, upper);
                cr_lower = vrhadd_u8(cr_lower, lower);
            }
            put_chroma::<W, false>(cb, cb_upper);
            put_chroma::<W, false>(cb.add(dst_stride), cb_lower);
            put_chroma::<W, false>(cr, cr_upper);
            put_chroma::<W, false>(cr.add(dst_stride), cr_lower);
            cb = cb.add(2 * dst_stride);
            cr = cr.add(2 * dst_stride);
        }
    }
}

/// The routine for a block in both chroma planes from one or two sources.
type ChromaBothFn = unsafe fn((*mut u8, *mut u8), usize, &PlaneSources, &PlaneSources, usize);

macro_rules! both_row {
    ($w:literal, $a:literal) => {
        [
            chroma_both::<$w, $a, 0>,
            chroma_both::<$w, $a, 1>,
            chroma_both::<$w, $a, 2>,
            chroma_both::<$w, $a, 3>,
            chroma_both::<$w, $a, NONE>,
        ]
    };
}

/// [`chroma_both`] by width (4 then 8), then the first source's kind of
/// offsets, then the second's, [`NONE`] last.
static CHROMA_BOTH: [[[ChromaBothFn; 5]; 4]; 2] = [
    [
        both_row!(4, 0),
        both_row!(4, 1),
        both_row!(4, 2),
        both_row!(4, 3),
    ],
    [
        both_row!(8, 0),
        both_row!(8, 1),
        both_row!(8, 2),
        both_row!(8, 3),
    ],
];

/// Which zero parts a chroma offset has, as [`chroma`]'s `MODE`.
fn chroma_mode((xf, yf): (u8, u8)) -> usize {
    usize::from(xf != 0) | usize::from(yf != 0) << 1
}

/// See [`crate::mc::chroma_planes`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn chroma_planes(
    dst: (&mut [u8], &mut [u8]),
    dst_stride: usize,
    size: (usize, usize),
    first: ChromaSource<'_>,
    second: Option<ChromaSource<'_>>,
) -> bool {
    let (dst_cb, dst_cr) = dst;
    if size.1 > 8 || dst_cb.len() != dst_cr.len() {
        return false;
    }
    // Each plane's source and the block's rows are checked as
    // `chroma_reach` checks one, and the Cr plane is as long as the Cb.
    let reach = |source: &ChromaSource<'_>| {
        if source.cb.len() != source.cr.len() {
            return None;
        }
        let cb = chroma_reach(&source.plane(true, size), source.frac, (dst_cb, dst_stride))?;
        let at = source.at.1 * source.stride + source.at.0;
        Some(PlaneSources {
            cb,
            cr: source.cr[at..].as_ptr(),
            stride: source.stride,
            frac: source.frac,
        })
    };
    let Some(one) = reach(&first) else {
        return false;
    };
    let (two, kind) = match second {
        Some(source) => match reach(&source) {
            Some(two) => (two, chroma_mode(source.frac)),
            None => return false,
        },
        None => (one, usize::from(NONE)),
    };
    let routine = CHROMA_BOTH[usize::from(size.0 == 8)][chroma_mode(first.frac)][kind];
    // SAFETY: `chroma_reach` checked every byte each plane's routine reads
    // from either source and writes, the Cr plane and block being as long
    // as the Cb ones.
    unsafe {
        routine(
            (dst_cb.as_mut_ptr(), dst_cr.as_mut_ptr()),
            dst_stride,
            &one,
            &two,
            size.1,
        );
    }
    true
}
