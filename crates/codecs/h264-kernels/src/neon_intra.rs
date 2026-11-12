//! The NEON whole-block intra predictions of [`crate::intra`]. The block,
//! the row above it and the column to its left are found in the plane with
//! one check, as rows that start one sample left of the block; each row is
//! then a fixed-size array.
//!
//! The plane mode is worked out in 16-bit lanes, as FFmpeg does: from
//! samples of 0 to 255, every sum before the shift lies between about
//! −11,500 and 19,700, so none overflows.

use core::arch::aarch64::{
    int16x8_t, uint8x16_t, vaddlvq_u8, vaddq_s16, vcombine_u8, vdupq_n_s16, vdupq_n_u8,
    vget_lane_u64, vld1q_s16, vqshrun_n_s16, vreinterpret_u64_u8,
};

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
