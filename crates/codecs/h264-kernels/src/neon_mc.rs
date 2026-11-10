//! NEON inter prediction. As in [`crate::neon`], everything here is safe
//! Rust on fixed-size arrays.

use core::arch::aarch64::{
    int16x4_t, int16x8_t, uint8x8_t, uint8x16_t, vaddl_s16, vaddl_u8, vcombine_u8, vcombine_u16,
    vcreate_u8, vdup_n_u8, vget_high_s16, vget_high_u8, vget_lane_u64, vget_low_s16, vget_low_u8,
    vmlal_u8, vmlaq_n_s32, vmlaq_n_u16, vmlsq_n_s32, vmlsq_n_u16, vmull_u8, vqmovn_u16,
    vqrshrun_n_s16, vqrshrun_n_s32, vreinterpret_u64_u8, vreinterpretq_s16_u8,
    vreinterpretq_s16_u16, vreinterpretq_u8_s16, vrhaddq_u8, vrshrn_n_u16,
};

use crate::neon::{halves, load16, pair, store16};

#[target_feature(enable = "neon")]
fn load8(bytes: &[u8; 8]) -> uint8x8_t {
    vcreate_u8(u64::from_le_bytes(*bytes))
}

#[target_feature(enable = "neon")]
fn store8(vector: uint8x8_t) -> [u8; 8] {
    vget_lane_u64::<0>(vreinterpret_u64_u8(vector)).to_le_bytes()
}

/// Eight 16-bit numbers as a vector.
#[target_feature(enable = "neon")]
fn load_wide(values: &[i16; 8]) -> int16x8_t {
    let mut bytes = [0u8; 16];
    for (pair, value) in bytes.as_chunks_mut::<2>().0.iter_mut().zip(values) {
        *pair = value.to_le_bytes();
    }
    vreinterpretq_s16_u8(load16(&bytes))
}

#[target_feature(enable = "neon")]
fn store_wide(vector: int16x8_t) -> [i16; 8] {
    let mut bytes = [0u8; 16];
    store16(vreinterpretq_u8_s16(vector), &mut bytes);
    core::array::from_fn(|index| i16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]]))
}

/// The first and second eight of sixteen.
fn split<T: Copy>(line: &[T; 16]) -> ([T; 8], [T; 8]) {
    (
        core::array::from_fn(|index| line[index]),
        core::array::from_fn(|index| line[index + 8]),
    )
}

fn join<T: Copy + Default>(first: [T; 8], second: [T; 8]) -> [T; 16] {
    let mut out = [T::default(); 16];
    out[..8].copy_from_slice(&first);
    out[8..].copy_from_slice(&second);
    out
}

/// The six-tap sum of eight samples, a + f − 5·(b + e) + 20·(c + d). The
/// unsigned arithmetic wraps where the sum is negative, and read as signed
/// 16-bit it is the true value, which lies between −2550 and 10710.
#[target_feature(enable = "neon")]
fn six(taps: [uint8x8_t; 6]) -> int16x8_t {
    let [a, b, c, d, e, f] = taps;
    let sum = vmlaq_n_u16(vaddl_u8(a, f), vaddl_u8(c, d), 20);
    vreinterpretq_s16_u16(vmlsq_n_u16(sum, vaddl_u8(b, e), 5))
}

#[target_feature(enable = "neon")]
fn six16(taps: [&[u8; 16]; 6]) -> (int16x8_t, int16x8_t) {
    let v: [uint8x16_t; 6] = taps.map(|line| load16(line));
    (
        six(v.map(|x| vget_low_u8(x))),
        six(v.map(|x| vget_high_u8(x))),
    )
}

#[target_feature(enable = "neon")]
pub(crate) fn half16(taps: [&[u8; 16]; 6]) -> [u8; 16] {
    let (first, second) = six16(taps);
    let mut out = [0u8; 16];
    store16(
        vcombine_u8(vqrshrun_n_s16::<5>(first), vqrshrun_n_s16::<5>(second)),
        &mut out,
    );
    out
}

#[target_feature(enable = "neon")]
pub(crate) fn half8(taps: [&[u8; 8]; 6]) -> [u8; 8] {
    store8(vqrshrun_n_s16::<5>(six(taps.map(|line| load8(line)))))
}

#[target_feature(enable = "neon")]
pub(crate) fn raw16(taps: [&[u8; 16]; 6]) -> [i16; 16] {
    let (first, second) = six16(taps);
    join(store_wide(first), store_wide(second))
}

#[target_feature(enable = "neon")]
pub(crate) fn raw8(taps: [&[u8; 8]; 6]) -> [i16; 8] {
    store_wide(six(taps.map(|line| load8(line))))
}

/// The six-tap sum of four first-pass sums in 32 bits, rounded, shifted by
/// 10 and clipped to 0..=65535.
#[target_feature(enable = "neon")]
fn centre4(taps: [int16x4_t; 6]) -> core::arch::aarch64::uint16x4_t {
    let [a, b, c, d, e, f] = taps;
    let sum = vmlaq_n_s32(vaddl_s16(a, f), vaddl_s16(c, d), 20);
    vqrshrun_n_s32::<10>(vmlsq_n_s32(sum, vaddl_s16(b, e), 5))
}

#[target_feature(enable = "neon")]
pub(crate) fn centre8(taps: [&[i16; 8]; 6]) -> [u8; 8] {
    let v: [int16x8_t; 6] = taps.map(|line| load_wide(line));
    let first = centre4(v.map(|x| vget_low_s16(x)));
    let second = centre4(v.map(|x| vget_high_s16(x)));
    // The narrowing clips to a sample.
    store8(vqmovn_u16(vcombine_u16(first, second)))
}

#[target_feature(enable = "neon")]
pub(crate) fn centre16(taps: [&[i16; 16]; 6]) -> [u8; 16] {
    let parts = taps.map(split);
    let [a, b, c, d, e, f] = &parts;
    join(
        centre8([&a.0, &b.0, &c.0, &d.0, &e.0, &f.0]),
        centre8([&a.1, &b.1, &c.1, &d.1, &e.1, &f.1]),
    )
}

#[target_feature(enable = "neon")]
pub(crate) fn avg16(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    store16(vrhaddq_u8(load16(a), load16(b)), &mut out);
    out
}

#[target_feature(enable = "neon")]
pub(crate) fn avg8(a: &[u8; 8], b: &[u8; 8]) -> [u8; 8] {
    store8(core::arch::aarch64::vrhadd_u8(load8(a), load8(b)))
}

#[target_feature(enable = "neon")]
pub(crate) fn chroma8(top: &[u8; 9], bottom: &[u8; 9], xf: u8, yf: u8) -> [u8; 8] {
    let (xf, yf) = (xf.min(7), yf.min(7));
    // Samples 0..8 in the low half and 1..9 in the high half.
    let both = |line: &[u8; 9]| {
        let left: [u8; 8] = core::array::from_fn(|index| line[index]);
        let right: [u8; 8] = core::array::from_fn(|index| line[index + 1]);
        pair(u64::from_le_bytes(left), u64::from_le_bytes(right))
    };
    let (above, below) = (both(top), both(bottom));
    // Each weight is at most 64 and their sum is 64, so the total fits 16
    // bits and the rounded shift by 6 is a sample.
    let mut sum = vmull_u8(vget_low_u8(above), vdup_n_u8((8 - xf) * (8 - yf)));
    sum = vmlal_u8(sum, vget_high_u8(above), vdup_n_u8(xf * (8 - yf)));
    sum = vmlal_u8(sum, vget_low_u8(below), vdup_n_u8((8 - xf) * yf));
    sum = vmlal_u8(sum, vget_high_u8(below), vdup_n_u8(xf * yf));
    let (low, _) = halves(vcombine_u8(vrshrn_n_u16::<6>(sum), vdup_n_u8(0)));
    low
}
