//! The NEON version of each kernel. Everything here is safe Rust: a vector
//! is made from an array's bytes and written back to an array, and no
//! function takes a pointer. Each function is built for NEON, so only the
//! callers in this crate that promise NEON is present reach them.

use core::arch::aarch64::{
    int16x8_t, uint8x8_t, uint8x16_t, uint16x8_t, vabdq_s16, vabdq_u8, vaddl_u8, vaddq_s16,
    vaddq_u8, vaddq_u16, vaddw_u8, vandq_s16, vandq_u8, vandq_u16, vbicq_u8, vbslq_s16, vbslq_u8,
    vcgezq_s8, vcgezq_s16, vcltq_s16, vcltq_u8, vcombine_s8, vcombine_u8, vcombine_u64,
    vcreate_u64, vdup_n_u8, vdupq_n_s8, vdupq_n_s16, vdupq_n_u8, vget_high_u8, vget_low_u8,
    vgetq_lane_u64, vhaddq_u8, vmaxq_s8, vmaxq_s16, vmaxq_u8, vmaxvq_u8, vminq_s8, vminq_s16,
    vminq_u8, vmovl_u8, vnegq_s8, vnegq_s16, vqaddq_u8, vqmovun_s16, vqrshrn_n_s16, vqsubq_u8,
    vreinterpret_u8_u64, vreinterpretq_s8_u8, vreinterpretq_s16_u16, vreinterpretq_u8_s8,
    vreinterpretq_u8_u16, vreinterpretq_u8_u32, vreinterpretq_u8_u64, vreinterpretq_u16_u8,
    vreinterpretq_u32_u8, vreinterpretq_u64_u8, vrhaddq_u8, vrshrn_n_u16, vshlq_n_s16, vshlq_n_u16,
    vshrq_n_s16, vsubl_u8, vsubq_s16, vsubq_u8, vtrn1q_u8, vtrn1q_u16, vtrn1q_u32, vtrn2q_u8,
    vtrn2q_u16, vtrn2q_u32, vuzp1q_u8, vuzp2q_u8, vzip1_u8, vzip1q_u64, vzip2_u8, vzip2q_u64,
};

use crate::deblock::{ChromaRows, Lines, Rows};

pub(crate) use crate::neon_mc::{
    avg_row, avg8, avg16, centre8, centre16, chroma8, half8, half16, raw8, raw16,
};
pub(crate) use crate::neon_qpel::{
    avg_block, chroma_block, chroma_pair, luma_block, luma_pair, weight_block,
};
pub(crate) use crate::neon_residual::{dc_add, idct4_add};

/// A vector of the 16 samples in `bytes`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn load16(bytes: &[u8; 16]) -> uint8x16_t {
    let value = u128::from_le_bytes(*bytes);
    pair(value as u64, (value >> 64) as u64)
}

/// A vector whose low half is `low`'s bytes and whose high half is `high`'s.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn pair(low: u64, high: u64) -> uint8x16_t {
    vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(low), vcreate_u64(high)))
}

/// The two halves of `vector` as bytes.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn halves(vector: uint8x16_t) -> ([u8; 8], [u8; 8]) {
    let wide = vreinterpretq_u64_u8(vector);
    (
        vgetq_lane_u64::<0>(wide).to_le_bytes(),
        vgetq_lane_u64::<1>(wide).to_le_bytes(),
    )
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn store16(vector: uint8x16_t, bytes: &mut [u8; 16]) {
    let (low, high) = halves(vector);
    bytes[..8].copy_from_slice(&low);
    bytes[8..].copy_from_slice(&high);
}

/// Turns the 8×8 block in the low halves of `v`, and the one in the high
/// halves, so rows become columns. Doing it twice gives `v` back.
/// A macro, so it is part of each function that uses it.
macro_rules! turn {
    ($v:expr) => {{
        let v: [uint8x16_t; 8] = $v;
        // Pairs of bytes, then pairs of 16 bits, then pairs of 32 bits. Each
        // step stays inside its own half of the vector.
        let a0 = vreinterpretq_u16_u8(vtrn1q_u8(v[0], v[1]));
        let a1 = vreinterpretq_u16_u8(vtrn2q_u8(v[0], v[1]));
        let a2 = vreinterpretq_u16_u8(vtrn1q_u8(v[2], v[3]));
        let a3 = vreinterpretq_u16_u8(vtrn2q_u8(v[2], v[3]));
        let a4 = vreinterpretq_u16_u8(vtrn1q_u8(v[4], v[5]));
        let a5 = vreinterpretq_u16_u8(vtrn2q_u8(v[4], v[5]));
        let a6 = vreinterpretq_u16_u8(vtrn1q_u8(v[6], v[7]));
        let a7 = vreinterpretq_u16_u8(vtrn2q_u8(v[6], v[7]));
        let wide = |x: uint16x8_t| vreinterpretq_u32_u8(vreinterpretq_u8_u16(x));
        let b0 = wide(vtrn1q_u16(a0, a2));
        let b1 = wide(vtrn1q_u16(a1, a3));
        let b2 = wide(vtrn2q_u16(a0, a2));
        let b3 = wide(vtrn2q_u16(a1, a3));
        let b4 = wide(vtrn1q_u16(a4, a6));
        let b5 = wide(vtrn1q_u16(a5, a7));
        let b6 = wide(vtrn2q_u16(a4, a6));
        let b7 = wide(vtrn2q_u16(a5, a7));
        [
            vreinterpretq_u8_u32(vtrn1q_u32(b0, b4)),
            vreinterpretq_u8_u32(vtrn1q_u32(b1, b5)),
            vreinterpretq_u8_u32(vtrn1q_u32(b2, b6)),
            vreinterpretq_u8_u32(vtrn1q_u32(b3, b7)),
            vreinterpretq_u8_u32(vtrn2q_u32(b0, b4)),
            vreinterpretq_u8_u32(vtrn2q_u32(b1, b5)),
            vreinterpretq_u8_u32(vtrn2q_u32(b2, b6)),
            vreinterpretq_u8_u32(vtrn2q_u32(b3, b7)),
        ]
    }};
}

#[target_feature(enable = "neon")]
fn load_rows(rows: &Rows) -> [uint8x16_t; 8] {
    [
        load16(&rows[0]),
        load16(&rows[1]),
        load16(&rows[2]),
        load16(&rows[3]),
        load16(&rows[4]),
        load16(&rows[5]),
        load16(&rows[6]),
        load16(&rows[7]),
    ]
}

/// Lines 0 to 7 fill the low halves and lines 8 to 15 the high halves, then
/// each half is turned, so vector `n` holds sample `n` of all sixteen lines.
#[target_feature(enable = "neon")]
fn load_lines(lines: &Lines) -> [uint8x16_t; 8] {
    let row = |index: usize| {
        pair(
            u64::from_le_bytes(lines[index]),
            u64::from_le_bytes(lines[index + 8]),
        )
    };
    turn!([
        row(0),
        row(1),
        row(2),
        row(3),
        row(4),
        row(5),
        row(6),
        row(7),
    ])
}

/// The inverse of [`load_lines`].
#[target_feature(enable = "neon")]
fn store_lines(columns: [uint8x16_t; 8], lines: &mut Lines) {
    let rows = turn!(columns);
    for (index, row) in rows.into_iter().enumerate() {
        let (low, high) = halves(row);
        lines[index] = low;
        lines[index + 8] = high;
    }
}

/// The low eight samples of `v` as 16-bit numbers.
#[target_feature(enable = "neon")]
fn low(v: uint8x16_t) -> int16x8_t {
    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v)))
}

/// A sample is at most 255 and a threshold fits 16 bits with room to spare.
fn narrow(value: i32) -> i16 {
    value.clamp(0, i32::from(i16::MAX)) as i16
}

/// 8.7.2.3 on sixteen lines, in the samples' own 8 bits. `v` is p2, p1,
/// p0, q0, q1, q2. Gives the new p1, p0, q0, q1, or `None` when no line
/// passes the thresholds and nothing changes. `tc0` is per four lines, at
/// most 125; a negative one leaves its lines alone.
/// A macro, so it is part of each function that uses it.
macro_rules! normal {
    ($v:expr, $alpha:expr, $beta:expr, $tc0:expr $(,)?) => {
        'filter: {
            let v: [uint8x16_t; 6] = $v;
            let (alpha, beta): (u8, u8) = ($alpha, $beta);
            let tc0: [i8; 4] = $tc0;
            let [p2, p1, p0, q0, q1, q2] = v;
            let alpha = vdupq_n_u8(alpha);
            let beta = vdupq_n_u8(beta);
            // Each clipping value four times over.
            let spread = |a: i8, b: i8| {
                u64::from(a as u8) * 0x0101_0101 | (u64::from(b as u8) * 0x0101_0101) << 32
            };
            let signed = vreinterpretq_s8_u8(pair(spread(tc0[0], tc0[1]), spread(tc0[2], tc0[3])));
            let filtered = vandq_u8(
                vandq_u8(
                    vcltq_u8(vabdq_u8(p0, q0), alpha),
                    vcltq_u8(vabdq_u8(p1, p0), beta),
                ),
                vandq_u8(vcltq_u8(vabdq_u8(q1, q0), beta), vcgezq_s8(signed)),
            );
            if vmaxvq_u8(filtered) == 0 {
                break 'filter None;
            }
            let ap = vandq_u8(vcltq_u8(vabdq_u8(p2, p0), beta), filtered);
            let aq = vandq_u8(vcltq_u8(vabdq_u8(q2, q0), beta), filtered);
            // Zero on a line that is not filtered, so nothing moves there.
            let tc0 = vandq_u8(vreinterpretq_u8_s8(signed), filtered);
            // A true lane is 0xff, which is −1, so subtracting it adds one.
            let tc = vreinterpretq_s8_u8(vsubq_u8(vsubq_u8(tc0, ap), aq));
            // p1 + Clip3(−tc0, tc0, (p2 + ((p0 + q0 + 1) >> 1) − 2·p1) >> 1) is the
            // halved sum of p2 and the average, held within tc0 of p1.
            let average = vrhaddq_u8(p0, q0);
            let side = |outer: uint8x16_t, inner: uint8x16_t, near: uint8x16_t| {
                let moved = vmaxq_u8(
                    vminq_u8(vhaddq_u8(outer, average), vqaddq_u8(inner, tc0)),
                    vqsubq_u8(inner, tc0),
                );
                vbslq_u8(near, moved, inner)
            };
            // (4·(q0 − p0) + (p1 − q1) + 4) >> 3 needs 16 bits on the way. It is
            // narrowed with saturation, which the clip to ±tc then hides.
            let half = |p1: uint8x8_t, p0: uint8x8_t, q0: uint8x8_t, q1: uint8x8_t| {
                let gap = vreinterpretq_s16_u16(vsubl_u8(q0, p0));
                let lean = vreinterpretq_s16_u16(vsubl_u8(p1, q1));
                vqrshrn_n_s16::<3>(vaddq_s16(vshlq_n_s16::<2>(gap), lean))
            };
            let raw = vcombine_s8(
                half(
                    vget_low_u8(p1),
                    vget_low_u8(p0),
                    vget_low_u8(q0),
                    vget_low_u8(q1),
                ),
                half(
                    vget_high_u8(p1),
                    vget_high_u8(p0),
                    vget_high_u8(q0),
                    vget_high_u8(q1),
                ),
            );
            let delta = vmaxq_s8(vminq_s8(raw, tc), vnegq_s8(tc));
            // Adding a signed step to a sample and holding it to 0..=255: add the
            // step's positive part, take away its negative part, both saturating.
            let zero = vdupq_n_s8(0);
            let rise = vreinterpretq_u8_s8(vmaxq_s8(delta, zero));
            let fall = vreinterpretq_u8_s8(vmaxq_s8(vnegq_s8(delta), zero));
            Some([
                side(p2, p1, ap),
                vqsubq_u8(vqaddq_u8(p0, rise), fall),
                vqaddq_u8(vqsubq_u8(q0, rise), fall),
                side(q2, q1, aq),
            ])
        }
    };
}

/// The thresholds as the 8-bit filters take them, when they fit: a sample
/// difference is at most 255, and a clipping value at most 125 leaves room
/// for the two that may be added to it.
fn small(alpha: i32, beta: i32, tc0: Option<[i8; 4]>) -> Option<(u8, u8)> {
    if tc0.is_some_and(|tc0| tc0.iter().any(|&group| group > 125)) {
        return None;
    }
    Some((u8::try_from(alpha).ok()?, u8::try_from(beta).ok()?))
}

#[target_feature(enable = "neon")]
pub(crate) fn luma_normal_rows(rows: &mut Rows, alpha: i32, beta: i32, tc0: [i8; 4]) {
    let Some((alpha, beta)) = small(alpha, beta, Some(tc0)) else {
        return crate::scalar::luma_normal_rows(rows, alpha, beta, tc0);
    };
    let v = load_rows(rows);
    let Some([p1, p0, q0, q1]) = normal!([v[1], v[2], v[3], v[4], v[5], v[6]], alpha, beta, tc0)
    else {
        return;
    };
    store16(p1, &mut rows[2]);
    store16(p0, &mut rows[3]);
    store16(q0, &mut rows[4]);
    store16(q1, &mut rows[5]);
}

#[target_feature(enable = "neon")]
pub(crate) fn luma_normal_lines(lines: &mut Lines, alpha: i32, beta: i32, tc0: [i8; 4]) {
    let Some((alpha, beta)) = small(alpha, beta, Some(tc0)) else {
        return crate::scalar::luma_normal_lines(lines, alpha, beta, tc0);
    };
    let mut v = load_lines(lines);
    let Some([p1, p0, q0, q1]) = normal!([v[1], v[2], v[3], v[4], v[5], v[6]], alpha, beta, tc0)
    else {
        return;
    };
    v[2] = p1;
    v[3] = p0;
    v[4] = q0;
    v[5] = q1;
    store_lines(v, lines);
}

#[target_feature(enable = "neon")]
pub(crate) fn luma_strong_rows(rows: &mut Rows, alpha: i32, beta: i32) {
    let Some((alpha, beta)) = small(alpha, beta, None) else {
        return crate::scalar::luma_strong_rows(rows, alpha, beta);
    };
    let v = load_rows(rows);
    let Some(out) = strong(v, alpha, beta) else {
        return;
    };
    for (row, vector) in rows.iter_mut().zip(out).skip(1).take(6) {
        store16(vector, row);
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn luma_strong_lines(lines: &mut Lines, alpha: i32, beta: i32) {
    let Some((alpha, beta)) = small(alpha, beta, None) else {
        return crate::scalar::luma_strong_lines(lines, alpha, beta);
    };
    let v = load_lines(lines);
    if let Some(out) = strong(v, alpha, beta) {
        store_lines(out, lines);
    }
}

/// 8.7.2.4 on sixteen lines. `v` is p3 to q3. Gives all eight back, p3 and
/// q3 as they came, or `None` when no line passes the thresholds and
/// nothing changes. The tests are made in the samples' own 8 bits; the
/// sums of up to eight samples take 16, eight lines at a time.
#[inline]
#[target_feature(enable = "neon")]
fn strong(v: [uint8x16_t; 8], alpha: u8, beta: u8) -> Option<[uint8x16_t; 8]> {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = v;
    let gap = vabdq_u8(p0, q0);
    let within = vdupq_n_u8(beta);
    let filtered = vandq_u8(
        vandq_u8(
            vcltq_u8(gap, vdupq_n_u8(alpha)),
            vcltq_u8(vabdq_u8(p1, p0), within),
        ),
        vcltq_u8(vabdq_u8(q1, q0), within),
    );
    if vmaxvq_u8(filtered) == 0 {
        return None;
    }
    // (alpha >> 2) + 2 is at most 65.
    let close = vandq_u8(filtered, vcltq_u8(gap, vdupq_n_u8((alpha >> 2) + 2)));
    // Eight lines of one side: `a0` to `a3` are its samples from the edge
    // outwards, and `b0`, `b1` the two nearest on the other side.
    let sums = |a: [uint8x8_t; 4], b0: uint8x8_t, b1: uint8x8_t| {
        let [a0, a1, a2, a3] = a;
        let three = vaddw_u8(vaddl_u8(a0, b0), a1);
        let four = vaddw_u8(three, a2);
        (
            // (a2 + 2·a1 + 2·a0 + 2·b0 + b1 + 4) >> 3
            vrshrn_n_u16::<3>(vaddw_u8(vaddq_u16(three, four), b1)),
            // (a2 + a1 + a0 + b0 + 2) >> 2
            vrshrn_n_u16::<2>(four),
            // (2·a3 + 3·a2 + a1 + a0 + b0 + 4) >> 3
            vrshrn_n_u16::<3>(vaddq_u16(vshlq_n_u16::<1>(vaddl_u8(a3, a2)), four)),
        )
    };
    // Gives the new a0, a1, a2.
    let side = |a: [uint8x16_t; 4], b0: uint8x16_t, b1: uint8x16_t| {
        let [a0, a1, a2, _] = a;
        let deep = vandq_u8(close, vcltq_u8(vabdq_u8(a2, a0), within));
        let (low0, low1, low2) = sums(a.map(|x| vget_low_u8(x)), vget_low_u8(b0), vget_low_u8(b1));
        let (high0, high1, high2) = sums(
            a.map(|x| vget_high_u8(x)),
            vget_high_u8(b0),
            vget_high_u8(b1),
        );
        // (2·a1 + a0 + b1 + 2) >> 2 is the rounded average of a1 and the
        // halved sum of a0 and b1.
        let weak0 = vrhaddq_u8(vhaddq_u8(a0, b1), a1);
        [
            vbslq_u8(
                filtered,
                vbslq_u8(deep, vcombine_u8(low0, high0), weak0),
                a0,
            ),
            vbslq_u8(deep, vcombine_u8(low1, high1), a1),
            vbslq_u8(deep, vcombine_u8(low2, high2), a2),
        ]
    };
    let [new_p0, new_p1, new_p2] = side([p0, p1, p2, p3], q0, q1);
    let [new_q0, new_q1, new_q2] = side([q0, q1, q2, q3], p0, p1);
    Some([p3, new_p2, new_p1, new_p0, new_q0, new_q1, new_q2, q3])
}

/// The eight samples in `bytes` as 16-bit numbers.
#[target_feature(enable = "neon")]
fn wide8(bytes: &[u8; 8]) -> int16x8_t {
    low(pair(u64::from_le_bytes(*bytes), 0))
}

/// The eight samples of `vector` as bytes.
#[target_feature(enable = "neon")]
fn bytes8(vector: uint8x8_t) -> [u8; 8] {
    halves(vcombine_u8(vector, vector)).0
}

/// Whether each chroma line is filtered at all: the three threshold tests.
#[target_feature(enable = "neon")]
fn chroma_filtered(v: [int16x8_t; 4], alpha: i32, beta: i32) -> uint16x8_t {
    let [p1, p0, q0, q1] = v;
    let alpha = vdupq_n_s16(narrow(alpha));
    let beta = vdupq_n_s16(narrow(beta));
    vandq_u16(
        vandq_u16(
            vcltq_s16(vabdq_s16(p0, q0), alpha),
            vcltq_s16(vabdq_s16(p1, p0), beta),
        ),
        vcltq_s16(vabdq_s16(q1, q0), beta),
    )
}

#[target_feature(enable = "neon")]
pub(crate) fn chroma_normal(rows: &mut ChromaRows, alpha: i32, beta: i32, tc0: [i8; 4]) {
    let v = [
        wide8(&rows[0]),
        wide8(&rows[1]),
        wide8(&rows[2]),
        wide8(&rows[3]),
    ];
    let [p1, p0, q0, q1] = v;
    // Each `tc0` covers two samples.
    let mut clip = [0u8; 8];
    for (pair, value) in clip.as_chunks_mut::<2>().0.iter_mut().zip(tc0) {
        *pair = [value as u8; 2];
    }
    // Sign-extended: a negative one stays negative.
    let tc0 = vsubq_s16(
        vandq_s16(wide8(&clip), vdupq_n_s16(0x7f)),
        vandq_s16(wide8(&clip), vdupq_n_s16(0x80)),
    );
    let filtered = vandq_u16(chroma_filtered(v, alpha, beta), vcgezq_s16(tc0));
    let tc = vaddq_s16(tc0, vdupq_n_s16(1));
    let raw = vshrq_n_s16::<3>(vaddq_s16(
        vaddq_s16(vshlq_n_s16::<2>(vsubq_s16(q0, p0)), vsubq_s16(p1, q1)),
        vdupq_n_s16(4),
    ));
    let delta = vandq_s16(
        vmaxq_s16(vminq_s16(raw, tc), vnegq_s16(tc)),
        vreinterpretq_s16_u16(filtered),
    );
    rows[1] = bytes8(vqmovun_s16(vaddq_s16(p0, delta)));
    rows[2] = bytes8(vqmovun_s16(vsubq_s16(q0, delta)));
}

#[target_feature(enable = "neon")]
pub(crate) fn chroma_strong(rows: &mut ChromaRows, alpha: i32, beta: i32) {
    let v = [
        wide8(&rows[0]),
        wide8(&rows[1]),
        wide8(&rows[2]),
        wide8(&rows[3]),
    ];
    let [p1, p0, q0, q1] = v;
    let filtered = chroma_filtered(v, alpha, beta);
    // (2·near + own + far + 2) >> 2
    let weak = |near: int16x8_t, own: int16x8_t, far: int16x8_t| {
        let sum = vaddq_s16(
            vaddq_s16(vshlq_n_s16::<1>(near), vaddq_s16(own, far)),
            vdupq_n_s16(2),
        );
        bytes8(vqmovun_s16(vbslq_s16(filtered, vshrq_n_s16::<2>(sum), own)))
    };
    rows[1] = weak(p1, p0, q1);
    rows[2] = weak(q1, q0, p1);
}

/// Eight lines of four samples as four rows of eight.
#[target_feature(enable = "neon")]
fn chroma_lines(lines: &[[u8; 4]; 8]) -> ChromaRows {
    let quad = |index: usize| {
        u64::from(u32::from_le_bytes(lines[index]))
            | u64::from(u32::from_le_bytes(lines[index + 1])) << 32
    };
    let first = pair(quad(0), quad(2));
    let second = pair(quad(4), quad(6));
    // Samples 0 and 2 of every line, then samples 1 and 3.
    let even = vuzp1q_u8(first, second);
    let odd = vuzp2q_u8(first, second);
    [
        halves(vuzp1q_u8(even, even)).0,
        halves(vuzp1q_u8(odd, odd)).0,
        halves(vuzp2q_u8(even, even)).0,
        halves(vuzp2q_u8(odd, odd)).0,
    ]
}

crate::edge::edge_kernels!(pub(crate), #[target_feature(enable = "neon")]);

/// Splits rows of `$n` samples off the front of a plane window, `$down`
/// samples apart, one for each name, and gives them as an array. The
/// function it is used in returns `false` if the window is too short.
macro_rules! rows {
    ($window:expr, $down:expr, $n:literal, [$($row:ident),+]) => {{
        let mut rest: &mut [u8] = $window;
        $(
            let taken = core::mem::take(&mut rest);
            let (row, tail) = taken.split_at_mut($down.min(taken.len()));
            rest = tail;
            let Some($row) = row.first_chunk_mut::<$n>() else {
                return false;
            };
        )+
        let _ = rest;
        [$($row),+]
    }};
}

/// One luma edge on eight vectors, p3 to q3: the vectors after the filter
/// and which of them it changed, or `None` when it changed nothing. The
/// thresholds are ones [`small`] took.
macro_rules! filter8 {
    ($v:expr, $filter:expr) => {{
        let mut v: [uint8x16_t; 8] = $v;
        let filter: crate::edge::Filter = $filter;
        match filter.tc0 {
            Some(tc0) => normal!(
                [v[1], v[2], v[3], v[4], v[5], v[6]],
                filter.alpha,
                filter.beta,
                tc0,
            )
            .map(|[p1, p0, q0, q1]| {
                v[2] = p1;
                v[3] = p0;
                v[4] = q0;
                v[5] = q1;
                (v, 2usize..6)
            }),
            None => strong(v, filter.alpha, filter.beta).map(|out| (out, 1usize..7)),
        }
    }};
}

/// The low halves of two vectors, then their high halves.
#[inline]
#[target_feature(enable = "neon")]
fn zip_halves(a: uint8x16_t, b: uint8x16_t) -> (uint8x16_t, uint8x16_t) {
    let (a, b) = (vreinterpretq_u64_u8(a), vreinterpretq_u64_u8(b));
    (
        vreinterpretq_u8_u64(vzip1q_u64(a, b)),
        vreinterpretq_u8_u64(vzip2q_u64(a, b)),
    )
}

/// An inner edge, one filtered at a strength below 4: its q0 is
/// `$columns[$first + 4]`, and the six vectors from `$columns[$first + 1]`
/// are p2 to q2. The four the filter changes are put back.
macro_rules! inner_edge {
    ($columns:ident, $first:expr, $filter:expr) => {
        if let Some(crate::edge::Filter {
            alpha,
            beta,
            tc0: Some(tc0),
        }) = $filter
        {
            let from = [
                $columns[$first + 1],
                $columns[$first + 2],
                $columns[$first + 3],
                $columns[$first + 4],
                $columns[$first + 5],
                $columns[$first + 6],
            ];
            if let Some([p1, p0, q0, q1]) = normal!(from, alpha, beta, tc0) {
                $columns[$first + 2] = p1;
                $columns[$first + 3] = p0;
                $columns[$first + 4] = q0;
                $columns[$first + 5] = q1;
            }
        }
    };
}

/// The sixteen samples from the fifth of a row of twenty.
macro_rules! inside {
    ($row:expr) => {{
        let mut samples = [0u8; 16];
        samples.copy_from_slice(&$row[4..]);
        load16(&samples)
    }};
}

/// Whether a macroblock's edges are ones [`luma_mb`] filters itself: every
/// threshold is one the 8-bit filters take, and no edge inside the
/// macroblock is filtered at strength 4, which only its boundary can be.
fn plain_edges(edges: &crate::edge::MbEdges) -> bool {
    let fits = |filter: &Option<crate::edge::Filter>| match filter {
        Some(filter) => filter.tame(),
        None => true,
    };
    let inner = |filter: &Option<crate::edge::Filter>| match filter {
        Some(filter) => filter.tc0.is_some() && filter.tame(),
        None => true,
    };
    let [v0, v1, v2, v3] = &edges.vertical;
    let [h0, h1, h2, h3] = &edges.horizontal;
    fits(v0)
        && inner(v1)
        && inner(v2)
        && inner(v3)
        && fits(h0)
        && inner(h1)
        && inner(h2)
        && inner(h3)
}

/// Every edge of a 16×16 luma macroblock. Its rows are found in the plane
/// once. The edge on the macroblock's boundary is filtered first, on its
/// own; the three inside it are then filtered together from vectors that
/// stay in registers. For the edges that run down the picture those are
/// the macroblock's columns, from one turn of it.
#[target_feature(enable = "neon")]
pub(crate) fn luma_mb(plane: &mut [u8], edges: &crate::edge::MbEdges) -> bool {
    let crate::edge::MbEdges {
        at,
        down,
        vertical,
        horizontal,
    } = *edges;
    if !crate::edge::mb_fits(plane.len(), edges, 16, 4) {
        return false;
    }
    // The rows are taken from four samples before the macroblock and four
    // rows above it. At the picture's top row and its first macroblock
    // those are not there, and each edge is filtered on its own; so is
    // each edge of a macroblock this does not take.
    let above = down
        .checked_mul(4)
        .and_then(|rows| at.checked_sub(rows + 4));
    let Some(above) = above.filter(|_| plain_edges(edges)) else {
        crate::edge::each_edge(edges, 4, |edge| {
            luma_edge(plane, edge);
        });
        return true;
    };
    let [v0, v1, v2, v3] = vertical;
    if v0.is_some() || v1.is_some() || v2.is_some() || v3.is_some() {
        let Some(window) = plane.get_mut(at - 4..) else {
            return false;
        };
        // Each row from four samples before the macroblock to its end.
        let rows = rows!(
            window,
            down,
            20,
            [
                r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
            ]
        );
        if let Some(filter) = v0 {
            // The boundary: sixteen lines of the eight samples around it.
            let line = |row: &[u8; 20]| {
                let mut samples = [0u8; 8];
                samples.copy_from_slice(&row[..8]);
                u64::from_le_bytes(samples)
            };
            let v = turn!([
                pair(line(rows[0]), line(rows[8])),
                pair(line(rows[1]), line(rows[9])),
                pair(line(rows[2]), line(rows[10])),
                pair(line(rows[3]), line(rows[11])),
                pair(line(rows[4]), line(rows[12])),
                pair(line(rows[5]), line(rows[13])),
                pair(line(rows[6]), line(rows[14])),
                pair(line(rows[7]), line(rows[15])),
            ]);
            if let Some((v, _)) = filter8!(v, filter) {
                let put = |row: &mut [u8; 20], line: u64| {
                    row[..8].copy_from_slice(&line.to_le_bytes());
                };
                let [b0, b1, b2, b3, b4, b5, b6, b7] =
                    turn!(v).map(|line| vreinterpretq_u64_u8(line));
                put(rows[0], vgetq_lane_u64::<0>(b0));
                put(rows[1], vgetq_lane_u64::<0>(b1));
                put(rows[2], vgetq_lane_u64::<0>(b2));
                put(rows[3], vgetq_lane_u64::<0>(b3));
                put(rows[4], vgetq_lane_u64::<0>(b4));
                put(rows[5], vgetq_lane_u64::<0>(b5));
                put(rows[6], vgetq_lane_u64::<0>(b6));
                put(rows[7], vgetq_lane_u64::<0>(b7));
                put(rows[8], vgetq_lane_u64::<1>(b0));
                put(rows[9], vgetq_lane_u64::<1>(b1));
                put(rows[10], vgetq_lane_u64::<1>(b2));
                put(rows[11], vgetq_lane_u64::<1>(b3));
                put(rows[12], vgetq_lane_u64::<1>(b4));
                put(rows[13], vgetq_lane_u64::<1>(b5));
                put(rows[14], vgetq_lane_u64::<1>(b6));
                put(rows[15], vgetq_lane_u64::<1>(b7));
            }
        }
        if v1.is_some() || v2.is_some() || v3.is_some() {
            // The macroblock turned once: `c[n]` is its column `n` down
            // all sixteen rows.
            let upper = turn!([
                inside!(rows[0]),
                inside!(rows[1]),
                inside!(rows[2]),
                inside!(rows[3]),
                inside!(rows[4]),
                inside!(rows[5]),
                inside!(rows[6]),
                inside!(rows[7]),
            ]);
            let lower = turn!([
                inside!(rows[8]),
                inside!(rows[9]),
                inside!(rows[10]),
                inside!(rows[11]),
                inside!(rows[12]),
                inside!(rows[13]),
                inside!(rows[14]),
                inside!(rows[15]),
            ]);
            let mut c = [pair(0, 0); 16];
            (c[0], c[8]) = zip_halves(upper[0], lower[0]);
            (c[1], c[9]) = zip_halves(upper[1], lower[1]);
            (c[2], c[10]) = zip_halves(upper[2], lower[2]);
            (c[3], c[11]) = zip_halves(upper[3], lower[3]);
            (c[4], c[12]) = zip_halves(upper[4], lower[4]);
            (c[5], c[13]) = zip_halves(upper[5], lower[5]);
            (c[6], c[14]) = zip_halves(upper[6], lower[6]);
            (c[7], c[15]) = zip_halves(upper[7], lower[7]);
            inner_edge!(c, 0, v1);
            inner_edge!(c, 4, v2);
            inner_edge!(c, 8, v3);
            let (u0, l0) = zip_halves(c[0], c[8]);
            let (u1, l1) = zip_halves(c[1], c[9]);
            let (u2, l2) = zip_halves(c[2], c[10]);
            let (u3, l3) = zip_halves(c[3], c[11]);
            let (u4, l4) = zip_halves(c[4], c[12]);
            let (u5, l5) = zip_halves(c[5], c[13]);
            let (u6, l6) = zip_halves(c[6], c[14]);
            let (u7, l7) = zip_halves(c[7], c[15]);
            let upper = turn!([u0, u1, u2, u3, u4, u5, u6, u7]);
            let lower = turn!([l0, l1, l2, l3, l4, l5, l6, l7]);
            let put = |row: &mut [u8; 20], samples: uint8x16_t| {
                let mut bytes = [0u8; 16];
                store16(samples, &mut bytes);
                row[4..].copy_from_slice(&bytes);
            };
            put(rows[0], upper[0]);
            put(rows[1], upper[1]);
            put(rows[2], upper[2]);
            put(rows[3], upper[3]);
            put(rows[4], upper[4]);
            put(rows[5], upper[5]);
            put(rows[6], upper[6]);
            put(rows[7], upper[7]);
            put(rows[8], lower[0]);
            put(rows[9], lower[1]);
            put(rows[10], lower[2]);
            put(rows[11], lower[3]);
            put(rows[12], lower[4]);
            put(rows[13], lower[5]);
            put(rows[14], lower[6]);
            put(rows[15], lower[7]);
        }
    }
    let [h0, h1, h2, h3] = horizontal;
    if h0.is_some() || h1.is_some() || h2.is_some() || h3.is_some() {
        let Some(window) = plane.get_mut(above + 4..) else {
            return false;
        };
        // Twenty rows: four above the macroblock, then its sixteen.
        let rows = rows!(
            window,
            down,
            16,
            [
                r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17,
                r18, r19
            ]
        );
        if let Some(filter) = h0 {
            // The boundary: the four rows above it and the four below.
            let v = [
                load16(rows[0]),
                load16(rows[1]),
                load16(rows[2]),
                load16(rows[3]),
                load16(rows[4]),
                load16(rows[5]),
                load16(rows[6]),
                load16(rows[7]),
            ];
            if let Some((v, changed)) = filter8!(v, filter) {
                if changed.start == 1 {
                    store16(v[1], rows[1]);
                    store16(v[6], rows[6]);
                }
                store16(v[2], rows[2]);
                store16(v[3], rows[3]);
                store16(v[4], rows[4]);
                store16(v[5], rows[5]);
            }
        }
        if h1.is_some() || h2.is_some() || h3.is_some() {
            // `r[n]` is the macroblock's row `n`.
            let mut r = [
                load16(rows[4]),
                load16(rows[5]),
                load16(rows[6]),
                load16(rows[7]),
                load16(rows[8]),
                load16(rows[9]),
                load16(rows[10]),
                load16(rows[11]),
                load16(rows[12]),
                load16(rows[13]),
                load16(rows[14]),
                load16(rows[15]),
                load16(rows[16]),
                load16(rows[17]),
                load16(rows[18]),
                load16(rows[19]),
            ];
            inner_edge!(r, 0, h1);
            inner_edge!(r, 4, h2);
            inner_edge!(r, 8, h3);
            // The filters change rows 2 to 13 at most.
            store16(r[2], rows[6]);
            store16(r[3], rows[7]);
            store16(r[4], rows[8]);
            store16(r[5], rows[9]);
            store16(r[6], rows[10]);
            store16(r[7], rows[11]);
            store16(r[8], rows[12]);
            store16(r[9], rows[13]);
            store16(r[10], rows[14]);
            store16(r[11], rows[15]);
            store16(r[12], rows[16]);
            store16(r[13], rows[17]);
        }
    }
    true
}

/// What two chroma edges filtered side by side need, the first edge in the
/// low eight lanes and the second in the high eight: alpha, beta, 0xff in
/// the lanes of an edge filtered at strength 4, and the clipping value of
/// each lane. An edge that is not filtered has an alpha of 0, which no
/// sample passes. The thresholds are ones [`small`] took.
#[inline]
#[target_feature(enable = "neon")]
fn chroma_thresholds(
    first: Option<crate::edge::Filter>,
    second: Option<crate::edge::Filter>,
) -> [uint8x16_t; 4] {
    // Alpha, beta, 0xff for strength 4, and the four clipping values.
    let half = |edge: Option<crate::edge::Filter>| -> (u8, u8, u8, u32) {
        match edge {
            None => (0, 0, 0, 0),
            Some(crate::edge::Filter {
                alpha,
                beta,
                tc0: None,
            }) => (alpha, beta, 0xff, 0),
            Some(crate::edge::Filter {
                alpha,
                beta,
                tc0: Some(tc0),
            }) => (
                alpha,
                beta,
                0,
                u32::from_le_bytes(tc0.map(|value| value as u8)),
            ),
        }
    };
    let (low, high) = (half(first), half(second));
    let both = |low: u8, high: u8| vcombine_u8(vdup_n_u8(low), vdup_n_u8(high));
    // Each clipping value covers two samples: the eight bytes zipped with
    // themselves.
    let eight = vreinterpret_u8_u64(vcreate_u64(u64::from(low.3) | u64::from(high.3) << 32));
    [
        both(low.0, high.0),
        both(low.1, high.1),
        both(low.2, high.2),
        vcombine_u8(vzip1_u8(eight, eight), vzip2_u8(eight, eight)),
    ]
}

/// The chroma filters of 8.7.2.3 and 8.7.2.4 on sixteen lanes, in the
/// samples' own 8 bits, each lane at the strength [`chroma_thresholds`]
/// gave it. `v` is p1, p0, q0, q1. Gives the new p0 and q0, or `None` when
/// no lane passes the thresholds and nothing changes.
/// A macro, so it is part of the function that uses it.
macro_rules! chroma_both {
    ($v:expr, $thresholds:expr) => {
        'filter: {
            let v: [uint8x16_t; 4] = $v;
            let thresholds: [uint8x16_t; 4] = $thresholds;
            let [p1, p0, q0, q1] = v;
            let [alpha, beta, strong, tc0] = thresholds;
            let filtered = vandq_u8(
                vandq_u8(
                    vcltq_u8(vabdq_u8(p0, q0), alpha),
                    vcltq_u8(vabdq_u8(p1, p0), beta),
                ),
                vcltq_u8(vabdq_u8(q1, q0), beta),
            );
            if vmaxvq_u8(filtered) == 0 {
                break 'filter None;
            }
            // Below strength 4: the lanes that are filtered, not at strength 4, and
            // whose clipping value is not negative. There tc is tc0 + 1; elsewhere
            // it is zero, so nothing moves.
            let below = vandq_u8(
                vbicq_u8(filtered, strong),
                vcgezq_s8(vreinterpretq_s8_u8(tc0)),
            );
            let tc = vreinterpretq_s8_u8(vandq_u8(vaddq_u8(tc0, vdupq_n_u8(1)), below));
            // (4·(q0 − p0) + (p1 − q1) + 4) >> 3 needs 16 bits on the way. It is
            // narrowed with saturation, which the clip to ±tc then hides.
            let half = |p1: uint8x8_t, p0: uint8x8_t, q0: uint8x8_t, q1: uint8x8_t| {
                let gap = vreinterpretq_s16_u16(vsubl_u8(q0, p0));
                let lean = vreinterpretq_s16_u16(vsubl_u8(p1, q1));
                vqrshrn_n_s16::<3>(vaddq_s16(vshlq_n_s16::<2>(gap), lean))
            };
            let raw = vcombine_s8(
                half(
                    vget_low_u8(p1),
                    vget_low_u8(p0),
                    vget_low_u8(q0),
                    vget_low_u8(q1),
                ),
                half(
                    vget_high_u8(p1),
                    vget_high_u8(p0),
                    vget_high_u8(q0),
                    vget_high_u8(q1),
                ),
            );
            let delta = vmaxq_s8(vminq_s8(raw, tc), vnegq_s8(tc));
            let zero = vdupq_n_s8(0);
            let rise = vreinterpretq_u8_s8(vmaxq_s8(delta, zero));
            let fall = vreinterpretq_u8_s8(vmaxq_s8(vnegq_s8(delta), zero));
            let p0_below = vqsubq_u8(vqaddq_u8(p0, rise), fall);
            let q0_below = vqaddq_u8(vqsubq_u8(q0, rise), fall);
            // At strength 4: (2·p1 + p0 + q1 + 2) >> 2, which is the rounded
            // average of p1 and the halved sum of p0 and q1.
            let hard = vandq_u8(filtered, strong);
            let p0_hard = vrhaddq_u8(vhaddq_u8(p0, q1), p1);
            let q0_hard = vrhaddq_u8(vhaddq_u8(q0, p1), q1);
            Some([
                vbslq_u8(hard, p0_hard, p0_below),
                vbslq_u8(hard, q0_hard, q0_below),
            ])
        }
    };
}

/// Every edge of an 8×8 chroma macroblock. Its rows are found in the plane
/// once, and the two edges of each direction, which share no sample, are
/// filtered side by side in one vector.
#[target_feature(enable = "neon")]
pub(crate) fn chroma_mb(plane: &mut [u8], edges: &crate::edge::MbEdges) -> bool {
    let crate::edge::MbEdges {
        at,
        down,
        vertical,
        horizontal,
    } = *edges;
    if !crate::edge::mb_fits(plane.len(), edges, 8, 2) {
        return false;
    }
    let [v0, v1, v2, v3] = vertical;
    let [h0, h1, h2, h3] = horizontal;
    // This takes the edges of 4:2:0 chroma, at 0 and 4, with thresholds
    // the 8-bit filter takes, in a macroblock with two rows above it and
    // two samples before it. Anything else is filtered edge by edge.
    let fits = |filter: &Option<crate::edge::Filter>| match filter {
        Some(filter) => filter.tame(),
        None => true,
    };
    let plain = v1.is_none()
        && v3.is_none()
        && h1.is_none()
        && h3.is_none()
        && fits(&v0)
        && fits(&v2)
        && fits(&h0)
        && fits(&h2);
    let above = down
        .checked_mul(2)
        .and_then(|rows| at.checked_sub(rows + 2));
    let Some(above) = above.filter(|_| plain) else {
        crate::edge::each_edge(edges, 2, |edge| {
            chroma_edge(plane, edge);
        });
        return true;
    };
    let Some(window) = plane.get_mut(above..) else {
        return false;
    };
    // Ten rows from two above the macroblock, each from two samples before
    // it: the macroblock's row `n` is `rows[n + 2]`, and its column `n` is
    // sample `n + 2` of a row.
    let rows = rows!(window, down, 10, [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9]);
    if v0.is_some() || v2.is_some() {
        let thresholds = chroma_thresholds(v0, v2);
        // The first eight samples of each of the macroblock's rows hold
        // both edges: p1, p0, q0, q1 of the one at 0, then of the one at 4.
        let line = |row: &[u8; 10]| {
            let mut samples = [0u8; 8];
            samples.copy_from_slice(&row[..8]);
            pair(u64::from_le_bytes(samples), 0)
        };
        // `c[n]` holds sample `n` of all eight lines in its low half.
        let c = turn!([
            line(rows[2]),
            line(rows[3]),
            line(rows[4]),
            line(rows[5]),
            line(rows[6]),
            line(rows[7]),
            line(rows[8]),
            line(rows[9]),
        ]);
        let (p1, _) = zip_halves(c[0], c[4]);
        let (p0, _) = zip_halves(c[1], c[5]);
        let (q0, _) = zip_halves(c[2], c[6]);
        let (q1, _) = zip_halves(c[3], c[7]);
        if let Some([p0, q0]) = chroma_both!([p1, p0, q0, q1], thresholds) {
            let (p0_first, p0_second) = zip_halves(p0, p0);
            let (q0_first, q0_second) = zip_halves(q0, q0);
            let back = turn!([
                c[0], p0_first, q0_first, c[3], c[4], p0_second, q0_second, c[7]
            ]);
            let put = |row: &mut [u8; 10], line: uint8x16_t| {
                let samples = vgetq_lane_u64::<0>(vreinterpretq_u64_u8(line));
                row[..8].copy_from_slice(&samples.to_le_bytes());
            };
            put(rows[2], back[0]);
            put(rows[3], back[1]);
            put(rows[4], back[2]);
            put(rows[5], back[3]);
            put(rows[6], back[4]);
            put(rows[7], back[5]);
            put(rows[8], back[6]);
            put(rows[9], back[7]);
        }
    }
    if h0.is_some() || h2.is_some() {
        let thresholds = chroma_thresholds(h0, h2);
        // The first eight rows hold both edges the same way.
        let row = |row: &[u8; 10]| {
            let mut samples = [0u8; 8];
            samples.copy_from_slice(&row[2..]);
            u64::from_le_bytes(samples)
        };
        let p1 = pair(row(rows[0]), row(rows[4]));
        let p0 = pair(row(rows[1]), row(rows[5]));
        let q0 = pair(row(rows[2]), row(rows[6]));
        let q1 = pair(row(rows[3]), row(rows[7]));
        if let Some([p0, q0]) = chroma_both!([p1, p0, q0, q1], thresholds) {
            let put = |row: &mut [u8; 10], samples: u64| {
                row[2..].copy_from_slice(&samples.to_le_bytes());
            };
            let (p0, q0) = (vreinterpretq_u64_u8(p0), vreinterpretq_u64_u8(q0));
            put(rows[1], vgetq_lane_u64::<0>(p0));
            put(rows[2], vgetq_lane_u64::<0>(q0));
            put(rows[5], vgetq_lane_u64::<1>(p0));
            put(rows[6], vgetq_lane_u64::<1>(q0));
        }
    }
    true
}
