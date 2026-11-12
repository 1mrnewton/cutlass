//! The NEON whole-block intra predictions of [`crate::intra`]. The block,
//! the row above it and the column to its left are found in the plane with
//! one check, as rows that start one sample left of the block; each row is
//! then a fixed-size array.
//!
//! The plane mode is worked out in 16-bit lanes, as FFmpeg does: from
//! samples of 0 to 255, every sum before the shift lies between about
//! −11,500 and 19,700, so none overflows.

use core::arch::aarch64::{
    int16x8_t, uint8x8_t, uint8x16_t, vaddlv_u8, vaddlvq_u8, vaddq_s16, vcombine_u8, vcreate_u8,
    vdup_n_u8, vdupq_n_s16, vdupq_n_u8, vext_u8, vextq_u8, vget_lane_u8, vget_lane_u64,
    vget_low_u8, vgetq_lane_u8, vhadd_u8, vhaddq_u8, vld1q_s16, vqshrun_n_s16, vreinterpret_u64_u8,
    vrev64_u8, vrhadd_u8, vrhaddq_u8,
};

use crate::intra::{Edges8, Near8, avg3, sample8};

use crate::neon::{load16, rows_of, store16};

/// A vector of the eight lanes of `lanes`.
#[inline]
#[target_feature(enable = "neon")]
fn load_lanes(lanes: &[i16; 8]) -> int16x8_t {
    // SAFETY: `lanes` is eight readable `i16`s, exactly what the load
    // reads, and it is aligned for `i16`.
    #[allow(unsafe_code)]
    unsafe {
        vld1q_s16(lanes.as_ptr())
    }
}

/// The plane prediction's gradient along `samples`, as in the plain twin:
/// the sample before the first is `corner`.
#[inline(always)]
fn gradient<const N: usize>(samples: &[u8; N], corner: u8) -> i32 {
    let half = N / 2;
    let mut sum = 0;
    for step in 0..half {
        let before = if step == half - 1 {
            corner
        } else {
            samples[half - 2 - step]
        };
        sum += (step as i32 + 1) * (i32::from(samples[half + step]) - i32::from(before));
    }
    sum
}

/// `a + b·(x − centre) + 16` for the eight columns from `first`.
#[inline(always)]
fn columns(a: i32, b: i32, first: i32, centre: i32) -> [i16; 8] {
    core::array::from_fn(|x| (a + b * (first + x as i32 - centre) + 16) as i16)
}

/// See [`crate::intra::pred16`].
#[target_feature(enable = "neon")]
pub(crate) fn pred16(
    plane: &mut [u8],
    at: usize,
    down: usize,
    mode: u8,
    near: (bool, bool),
) -> bool {
    let Some(start) = crate::intra::corner_at::<16>(plane.len(), at, down) else {
        return false;
    };
    let Some(window) = plane.get_mut(start..) else {
        return false;
    };
    let Some(rows) = rows_of::<17, 17>(window, down) else {
        return false;
    };
    let [top, body @ ..] = rows;
    let [corner, up @ ..] = &*top;
    let (corner, up) = (*corner, *up);
    let side: [u8; 16] = core::array::from_fn(|y| body[y][0]);
    // Each row of the block, the sixteen samples after its left neighbour.
    let mut put = |y: usize, samples: uint8x16_t| {
        let [_, row @ ..] = &mut *body[y];
        store16(samples, row);
    };
    let (above, left) = near;
    match mode {
        0 if above => {
            let samples = load16(&up);
            for y in 0..16 {
                put(y, samples);
            }
        }
        1 if left => {
            for (y, &sample) in side.iter().enumerate() {
                put(y, vdupq_n_u8(sample));
            }
        }
        2 => {
            let (sum_up, sum_side) = (vaddlvq_u8(load16(&up)), vaddlvq_u8(load16(&side)));
            let dc = match near {
                (true, true) => (u32::from(sum_up) + u32::from(sum_side) + 16) >> 5,
                (true, false) => (u32::from(sum_up) + 8) >> 4,
                (false, true) => (u32::from(sum_side) + 8) >> 4,
                (false, false) => 128,
            };
            let samples = vdupq_n_u8(dc as u8);
            for y in 0..16 {
                put(y, samples);
            }
        }
        3 if above && left => {
            let (h, v) = (gradient(&up, corner), gradient(&side, corner));
            let a = 16 * (i32::from(up[15]) + i32::from(side[15]));
            let (b, c) = ((5 * h + 32) >> 6, (5 * v + 32) >> 6);
            // The first row's sums before the shift, then a row's step.
            let mut low = load_lanes(&columns(a - 7 * c, b, 0, 7));
            let mut high = load_lanes(&columns(a - 7 * c, b, 8, 7));
            let step = vdupq_n_s16(c as i16);
            for y in 0..16 {
                put(
                    y,
                    vcombine_u8(vqshrun_n_s16::<5>(low), vqshrun_n_s16::<5>(high)),
                );
                low = vaddq_s16(low, step);
                high = vaddq_s16(high, step);
            }
        }
        _ => {
            let samples = vdupq_n_u8(128);
            for y in 0..16 {
                put(y, samples);
            }
        }
    }
    true
}

/// See [`crate::intra::pred_chroma`].
#[target_feature(enable = "neon")]
pub(crate) fn pred_chroma(
    plane: &mut [u8],
    at: usize,
    down: usize,
    mode: u8,
    near: (bool, bool),
) -> bool {
    let Some(start) = crate::intra::corner_at::<8>(plane.len(), at, down) else {
        return false;
    };
    let Some(window) = plane.get_mut(start..) else {
        return false;
    };
    let Some(rows) = rows_of::<9, 9>(window, down) else {
        return false;
    };
    let [top, body @ ..] = rows;
    let [corner, up @ ..] = &*top;
    // The row above as one word, so that a mode that copies it moves it
    // whole rather than byte by byte.
    let (corner, up_word) = (*corner, u64::from_le_bytes(*up));
    let up = up_word.to_le_bytes();
    let side: [u8; 8] = core::array::from_fn(|y| body[y][0]);
    let mut put = |y: usize, samples: u64| {
        let [_, row @ ..] = &mut *body[y];
        *row = samples.to_le_bytes();
    };
    let (above, left) = near;
    // Eight copies of a sample, or four of each of two.
    const ONES: u64 = 0x0101_0101_0101_0101;
    match mode {
        1 if left => {
            for (y, &sample) in side.iter().enumerate() {
                put(y, u64::from(sample) * ONES);
            }
        }
        2 if above => {
            for y in 0..8 {
                put(y, up_word);
            }
        }
        3 if above && left => {
            let (h, v) = (gradient(&up, corner), gradient(&side, corner));
            let a = 16 * (i32::from(up[7]) + i32::from(side[7]));
            let (b, c) = ((17 * h + 16) >> 5, (17 * v + 16) >> 5);
            let mut sums = load_lanes(&columns(a - 3 * c, b, 0, 3));
            let step = vdupq_n_s16(c as i16);
            for y in 0..8 {
                put(
                    y,
                    vget_lane_u64::<0>(vreinterpret_u64_u8(vqshrun_n_s16::<5>(sums))),
                );
                sums = vaddq_s16(sums, step);
            }
        }
        _ => {
            let quarter = |samples: &[u8; 8], half: usize| -> u32 {
                samples[half * 4..half * 4 + 4]
                    .iter()
                    .map(|&s| u32::from(s))
                    .sum()
            };
            let (up_left, up_right) = (quarter(&up, 0), quarter(&up, 1));
            let (side_top, side_bottom) = (quarter(&side, 0), quarter(&side, 1));
            // The DC of each 4×4 quarter: from both edges beside it when
            // the quarter touches both, else from the one edge it touches,
            // else from the other.
            let dc = |along: u32, beside: u32, diagonal: bool, top: bool| -> u64 {
                u64::from(match near {
                    (true, true) if diagonal => (along + beside + 4) >> 3,
                    (true, true) if top => (along + 2) >> 2,
                    (true, true) => (beside + 2) >> 2,
                    (true, false) => (along + 2) >> 2,
                    (false, true) => (beside + 2) >> 2,
                    (false, false) => 128,
                })
            };
            let halves = |left: u64, right: u64| (left | right << 32) * (ONES & 0xffff_ffff);
            let upper = halves(
                dc(up_left, side_top, true, true),
                dc(up_right, side_top, false, true),
            );
            let lower = halves(
                dc(up_left, side_bottom, false, false),
                dc(up_right, side_bottom, true, false),
            );
            for y in 0..8 {
                put(y, if y < 4 { upper } else { lower });
            }
        }
    }
    true
}

/// The (a + 2b + c + 2) / 4 of each lane: half the sum of the outer two,
/// then rounded halfway to the middle, which is the same.
#[inline]
#[target_feature(enable = "neon")]
fn avg3q(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
    vrhaddq_u8(vhaddq_u8(a, c), b)
}

/// [`avg3q`] on eight lanes.
#[inline]
#[target_feature(enable = "neon")]
fn avg3d(a: uint8x8_t, b: uint8x8_t, c: uint8x8_t) -> uint8x8_t {
    vrhadd_u8(vhadd_u8(a, c), b)
}

/// The eight lanes of `vector` as bytes.
#[inline]
#[target_feature(enable = "neon")]
fn bytes8(vector: uint8x8_t) -> [u8; 8] {
    vget_lane_u64::<0>(vreinterpret_u64_u8(vector)).to_le_bytes()
}

/// The sixteen lanes of `vector` as bytes.
#[inline]
#[target_feature(enable = "neon")]
fn bytes16(vector: uint8x16_t) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    store16(vector, &mut bytes);
    bytes
}

/// The eight bytes from lane `N` of the pair `low`, `high`.
macro_rules! window8 {
    ($low:expr, $high:expr, $n:literal) => {
        bytes8(vget_low_u8(vextq_u8::<$n>($low, $high)))
    };
}

/// See [`crate::intra::pred8`].
#[target_feature(enable = "neon")]
pub(crate) fn pred8(plane: &mut [u8], at: usize, down: usize, mode: u8, near: Near8) -> bool {
    let Some(start) = crate::intra::corner8_at(plane.len(), at, down) else {
        return false;
    };
    // The corner and the sixteen samples after it, copied out before the
    // block's rows are borrowed: the two may share bytes.
    let Some(&top) = plane.get(start..).and_then(|row| row.first_chunk::<17>()) else {
        return false;
    };
    let Some(window) = plane.get_mut(at - 1..) else {
        return false;
    };
    let Some(mut rows) = rows_of::<9, 8>(window, down) else {
        return false;
    };
    let [raw_corner, raw_up @ ..] = top;
    let raw_side = vcreate_u8(u64::from_le_bytes(core::array::from_fn(|y| rows[y][0])));
    // The sixteen samples above, the last eight repeating the eighth when
    // those above and right are not there; 128 when none are.
    let mut raw_above = load16(&raw_up);
    if !near.above_right {
        raw_above = vcombine_u8(vget_low_u8(raw_above), vdup_n_u8(raw_up[7]));
    }
    if !near.above {
        raw_above = vdupq_n_u8(128);
    }
    let raw_left = if near.left { raw_side } else { vdup_n_u8(128) };
    let corner = if near.corner { raw_corner } else { 128 };
    let first_above = vgetq_lane_u8::<0>(raw_above);
    let first_left = vget_lane_u8::<0>(raw_left);
    // The filter of clause 8.3.2.2.1, on edges that are there.
    let above = if near.above {
        let before = if near.corner { corner } else { first_above };
        let last = vdupq_n_u8(vgetq_lane_u8::<15>(raw_above));
        avg3q(
            vextq_u8::<15>(vdupq_n_u8(before), raw_above),
            raw_above,
            vextq_u8::<1>(raw_above, last),
        )
    } else {
        raw_above
    };
    let left = if near.left {
        let before = if near.corner { corner } else { first_left };
        let last = vdup_n_u8(vget_lane_u8::<7>(raw_left));
        avg3d(
            vext_u8::<7>(vdup_n_u8(before), raw_left),
            raw_left,
            vext_u8::<1>(raw_left, last),
        )
    } else {
        raw_left
    };
    let corner = match (near.corner, near.above, near.left) {
        (true, true, true) => avg3(first_above, corner, first_left),
        (true, true, false) => avg3(corner, corner, first_above),
        (true, false, true) => avg3(corner, corner, first_left),
        _ => corner,
    };
    let mut put = |y: usize, samples: [u8; 8]| {
        let [_, row @ ..] = &mut *rows[y];
        *row = samples;
    };
    let both = near.above && near.left;
    match mode {
        0 if near.above => {
            let samples = bytes8(vget_low_u8(above));
            for y in 0..8 {
                put(y, samples);
            }
        }
        1 if near.left => {
            for (y, sample) in bytes8(left).into_iter().enumerate() {
                put(y, [sample; 8]);
            }
        }
        3 if near.above => {
            // Each row is the filtered row above, filtered again, from its
            // own place: the last sample repeats past the end.
            let last = vdupq_n_u8(vgetq_lane_u8::<15>(above));
            let next = vextq_u8::<1>(above, last);
            let after = vextq_u8::<2>(above, last);
            let t = avg3q(above, next, after);
            put(0, window8!(t, last, 0));
            put(1, window8!(t, last, 1));
            put(2, window8!(t, last, 2));
            put(3, window8!(t, last, 3));
            put(4, window8!(t, last, 4));
            put(5, window8!(t, last, 5));
            put(6, window8!(t, last, 6));
            put(7, window8!(t, last, 7));
        }
        4 if both => {
            // The left column bottom up, the corner, then the row above:
            // each row is that line filtered again, from its own place.
            let line = vcombine_u8(
                vrev64_u8(left),
                vext_u8::<7>(vdup_n_u8(corner), vget_low_u8(above)),
            );
            let tail = vextq_u8::<7>(above, above);
            let d = avg3q(line, vextq_u8::<1>(line, tail), vextq_u8::<2>(line, tail));
            put(0, window8!(d, d, 7));
            put(1, window8!(d, d, 6));
            put(2, window8!(d, d, 5));
            put(3, window8!(d, d, 4));
            put(4, window8!(d, d, 3));
            put(5, window8!(d, d, 2));
            put(6, window8!(d, d, 1));
            put(7, window8!(d, d, 0));
        }
        7 if near.above => {
            let next = vextq_u8::<1>(above, above);
            let after = vextq_u8::<2>(above, above);
            let halves = vrhaddq_u8(above, next);
            let thirds = avg3q(above, next, after);
            put(0, window8!(halves, halves, 0));
            put(1, window8!(thirds, thirds, 0));
            put(2, window8!(halves, halves, 1));
            put(3, window8!(thirds, thirds, 1));
            put(4, window8!(halves, halves, 2));
            put(5, window8!(thirds, thirds, 2));
            put(6, window8!(halves, halves, 3));
            put(7, window8!(thirds, thirds, 3));
        }
        5 | 6 if both => {
            let edges = Edges8 {
                above: bytes16(above),
                left: bytes8(left),
                corner,
            };
            for y in 0..8 {
                put(y, core::array::from_fn(|x| sample8(mode, x, y, &edges)));
            }
        }
        8 if near.left => {
            let edges = Edges8 {
                above: bytes16(above),
                left: bytes8(left),
                corner,
            };
            for y in 0..8 {
                put(y, core::array::from_fn(|x| sample8(mode, x, y, &edges)));
            }
        }
        _ => {
            let sum_above = u32::from(vaddlv_u8(vget_low_u8(above)));
            let sum_left = u32::from(vaddlv_u8(left));
            let dc = match (near.above, near.left) {
                (true, true) => (sum_above + sum_left + 8) >> 4,
                (true, false) => (sum_above + 4) >> 3,
                (false, true) => (sum_left + 4) >> 3,
                (false, false) => 128,
            } as u8;
            for y in 0..8 {
                put(y, [dc; 8]);
            }
        }
    }
    true
}
