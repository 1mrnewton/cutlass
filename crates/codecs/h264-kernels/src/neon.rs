//! The NEON version of each kernel. Everything here is safe Rust: a vector
//! is made from an array's bytes and written back to an array, and no
//! function takes a pointer. Each function is built for NEON, so only the
//! callers in this crate that promise NEON is present reach them.

use core::arch::aarch64::{
    int16x8_t, uint8x8_t, uint8x16_t, uint16x8_t, vabdq_s16, vaddq_s16, vandq_s16, vandq_u16,
    vbslq_s16, vcgezq_s16, vcltq_s16, vcombine_s16, vcombine_u8, vcombine_u64, vcreate_u64,
    vdup_n_s16, vdupq_n_s16, vget_high_u8, vget_low_u8, vgetq_lane_u64, vmaxq_s16, vminq_s16,
    vmovl_u8, vnegq_s16, vqmovun_s16, vreinterpretq_s16_u16, vreinterpretq_u8_u16,
    vreinterpretq_u8_u32, vreinterpretq_u8_u64, vreinterpretq_u16_u8, vreinterpretq_u32_u8,
    vreinterpretq_u64_u8, vshlq_n_s16, vshrq_n_s16, vsubq_s16, vtrn1q_u8, vtrn1q_u16, vtrn1q_u32,
    vtrn2q_u8, vtrn2q_u16, vtrn2q_u32,
};

use crate::deblock::{ChromaRows, Lines, Rows};

pub(crate) use crate::neon_mc::{
    avg8, avg16, centre8, centre16, chroma8, half8, half16, raw8, raw16,
};

/// A vector of the 16 samples in `bytes`.
#[target_feature(enable = "neon")]
pub(crate) fn load16(bytes: &[u8; 16]) -> uint8x16_t {
    let value = u128::from_le_bytes(*bytes);
    pair(value as u64, (value >> 64) as u64)
}

/// A vector whose low half is `low`'s bytes and whose high half is `high`'s.
#[target_feature(enable = "neon")]
pub(crate) fn pair(low: u64, high: u64) -> uint8x16_t {
    vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(low), vcreate_u64(high)))
}

/// The two halves of `vector` as bytes.
#[target_feature(enable = "neon")]
pub(crate) fn halves(vector: uint8x16_t) -> ([u8; 8], [u8; 8]) {
    let wide = vreinterpretq_u64_u8(vector);
    (
        vgetq_lane_u64::<0>(wide).to_le_bytes(),
        vgetq_lane_u64::<1>(wide).to_le_bytes(),
    )
}

#[target_feature(enable = "neon")]
pub(crate) fn store16(vector: uint8x16_t, bytes: &mut [u8; 16]) {
    let (low, high) = halves(vector);
    bytes[..8].copy_from_slice(&low);
    bytes[8..].copy_from_slice(&high);
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
    turn([
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
    let rows = turn(columns);
    for (index, row) in rows.into_iter().enumerate() {
        let (low, high) = halves(row);
        lines[index] = low;
        lines[index + 8] = high;
    }
}

/// Turns the 8×8 block in the low halves of `v`, and the one in the high
/// halves, so rows become columns. Doing it twice gives `v` back.
#[target_feature(enable = "neon")]
fn turn(v: [uint8x16_t; 8]) -> [uint8x16_t; 8] {
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
}

/// The low eight samples of `v` as 16-bit numbers.
#[target_feature(enable = "neon")]
fn low(v: uint8x16_t) -> int16x8_t {
    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v)))
}

/// The high eight samples of `v` as 16-bit numbers.
#[target_feature(enable = "neon")]
fn high(v: uint8x16_t) -> int16x8_t {
    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v)))
}

/// A sample is at most 255 and a threshold fits 16 bits with room to spare.
fn narrow(value: i32) -> i16 {
    value.clamp(0, i32::from(i16::MAX)) as i16
}

#[target_feature(enable = "neon")]
pub(crate) fn luma_normal_rows(rows: &mut Rows, alpha: i32, beta: i32, tc0: [i8; 4]) {
    let v = load_rows(rows);
    let [p1, p0, q0, q1] = normal([v[1], v[2], v[3], v[4], v[5], v[6]], alpha, beta, tc0);
    store16(p1, &mut rows[2]);
    store16(p0, &mut rows[3]);
    store16(q0, &mut rows[4]);
    store16(q1, &mut rows[5]);
}

#[target_feature(enable = "neon")]
pub(crate) fn luma_normal_lines(lines: &mut Lines, alpha: i32, beta: i32, tc0: [i8; 4]) {
    let mut v = load_lines(lines);
    let [p1, p0, q0, q1] = normal([v[1], v[2], v[3], v[4], v[5], v[6]], alpha, beta, tc0);
    v[2] = p1;
    v[3] = p0;
    v[4] = q0;
    v[5] = q1;
    store_lines(v, lines);
}

#[target_feature(enable = "neon")]
pub(crate) fn luma_strong_rows(rows: &mut Rows, alpha: i32, beta: i32) {
    let v = load_rows(rows);
    let out = strong(v, alpha, beta);
    for (row, vector) in rows.iter_mut().zip(out).skip(1).take(6) {
        store16(vector, row);
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn luma_strong_lines(lines: &mut Lines, alpha: i32, beta: i32) {
    let v = load_lines(lines);
    store_lines(strong(v, alpha, beta), lines);
}

/// 8.7.2.3 on sixteen lines. `v` is p2, p1, p0, q0, q1, q2. Gives the new
/// p1, p0, q0, q1.
#[target_feature(enable = "neon")]
fn normal(v: [uint8x16_t; 6], alpha: i32, beta: i32, tc0: [i8; 4]) -> [uint8x16_t; 4] {
    let alpha = vdupq_n_s16(narrow(alpha));
    let beta = vdupq_n_s16(narrow(beta));
    let clip = |a: i8, b: i8| vcombine_s16(vdup_n_s16(i16::from(a)), vdup_n_s16(i16::from(b)));
    let first = normal8(v.map(|x| low(x)), alpha, beta, clip(tc0[0], tc0[1]));
    let second = normal8(v.map(|x| high(x)), alpha, beta, clip(tc0[2], tc0[3]));
    [
        vcombine_u8(first[0], second[0]),
        vcombine_u8(first[1], second[1]),
        vcombine_u8(first[2], second[2]),
        vcombine_u8(first[3], second[3]),
    ]
}

/// [`normal`] on eight lines, in 16-bit arithmetic. A negative `tc0` lane is
/// a line that is not filtered.
#[target_feature(enable = "neon")]
fn normal8(v: [int16x8_t; 6], alpha: int16x8_t, beta: int16x8_t, tc0: int16x8_t) -> [uint8x8_t; 4] {
    let [p2, p1, p0, q0, q1, q2] = v;
    let filtered = vandq_u16(
        vandq_u16(
            vcltq_s16(vabdq_s16(p0, q0), alpha),
            vcltq_s16(vabdq_s16(p1, p0), beta),
        ),
        vandq_u16(vcltq_s16(vabdq_s16(q1, q0), beta), vcgezq_s16(tc0)),
    );
    let ap = vcltq_s16(vabdq_s16(p2, p0), beta);
    let aq = vcltq_s16(vabdq_s16(q2, q0), beta);
    // A true lane is −1, so subtracting it adds one.
    let tc = vsubq_s16(
        vsubq_s16(tc0, vreinterpretq_s16_u16(ap)),
        vreinterpretq_s16_u16(aq),
    );
    let raw = vshrq_n_s16::<3>(vaddq_s16(
        vaddq_s16(vshlq_n_s16::<2>(vsubq_s16(q0, p0)), vsubq_s16(p1, q1)),
        vdupq_n_s16(4),
    ));
    let delta = vandq_s16(
        vmaxq_s16(vminq_s16(raw, tc), vnegq_s16(tc)),
        vreinterpretq_s16_u16(filtered),
    );
    let average = vshrq_n_s16::<1>(vaddq_s16(vaddq_s16(p0, q0), vdupq_n_s16(1)));
    let lower = vnegq_s16(tc0);
    let side = |outer: int16x8_t, inner: int16x8_t, near: uint16x8_t| {
        let step = vshrq_n_s16::<1>(vsubq_s16(
            vaddq_s16(outer, average),
            vshlq_n_s16::<1>(inner),
        ));
        let step = vmaxq_s16(vminq_s16(step, tc0), lower);
        let on = vreinterpretq_s16_u16(vandq_u16(filtered, near));
        vqmovun_s16(vaddq_s16(inner, vandq_s16(step, on)))
    };
    [
        side(p2, p1, ap),
        vqmovun_s16(vaddq_s16(p0, delta)),
        vqmovun_s16(vsubq_s16(q0, delta)),
        side(q2, q1, aq),
    ]
}

/// 8.7.2.4 on sixteen lines. `v` is p3 to q3. Gives all eight back; p3 and
/// q3 are as they came.
#[target_feature(enable = "neon")]
fn strong(v: [uint8x16_t; 8], alpha: i32, beta: i32) -> [uint8x16_t; 8] {
    let small = vdupq_n_s16(narrow((alpha >> 2) + 2));
    let alpha = vdupq_n_s16(narrow(alpha));
    let beta = vdupq_n_s16(narrow(beta));
    let first = strong8(v.map(|x| low(x)), alpha, beta, small);
    let second = strong8(v.map(|x| high(x)), alpha, beta, small);
    [
        v[0],
        vcombine_u8(first[0], second[0]),
        vcombine_u8(first[1], second[1]),
        vcombine_u8(first[2], second[2]),
        vcombine_u8(first[3], second[3]),
        vcombine_u8(first[4], second[4]),
        vcombine_u8(first[5], second[5]),
        v[7],
    ]
}

/// [`strong`] on eight lines. Gives the new p2, p1, p0, q0, q1, q2.
#[target_feature(enable = "neon")]
fn strong8(
    v: [int16x8_t; 8],
    alpha: int16x8_t,
    beta: int16x8_t,
    small: int16x8_t,
) -> [uint8x8_t; 6] {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = v;
    let gap = vabdq_s16(p0, q0);
    let filtered = vandq_u16(
        vandq_u16(vcltq_s16(gap, alpha), vcltq_s16(vabdq_s16(p1, p0), beta)),
        vcltq_s16(vabdq_s16(q1, q0), beta),
    );
    let close = vandq_u16(filtered, vcltq_s16(gap, small));
    // One side of the edge: `a` are its samples from the edge outwards, and
    // `b0`, `b1` the two nearest on the other side. Gives the new a0, a1, a2.
    let side = |a: [int16x8_t; 4], b0: int16x8_t, b1: int16x8_t| {
        let [a0, a1, a2, a3] = a;
        let deep = vandq_u16(close, vcltq_s16(vabdq_s16(a2, a0), beta));
        let sum = vaddq_s16(vaddq_s16(a0, b0), a1);
        // (a2 + 2·a1 + 2·a0 + 2·b0 + b1 + 4) >> 3
        let strong0 = vshrq_n_s16::<3>(vaddq_s16(
            vaddq_s16(vshlq_n_s16::<1>(sum), vaddq_s16(a2, b1)),
            vdupq_n_s16(4),
        ));
        // (a2 + a1 + a0 + b0 + 2) >> 2
        let strong1 = vshrq_n_s16::<2>(vaddq_s16(vaddq_s16(sum, a2), vdupq_n_s16(2)));
        // (2·a3 + 3·a2 + a1 + a0 + b0 + 4) >> 3
        let strong2 = vshrq_n_s16::<3>(vaddq_s16(
            vaddq_s16(vshlq_n_s16::<1>(vaddq_s16(a3, a2)), vaddq_s16(a2, sum)),
            vdupq_n_s16(4),
        ));
        // (2·a1 + a0 + b1 + 2) >> 2
        let weak0 = vshrq_n_s16::<2>(vaddq_s16(
            vaddq_s16(vshlq_n_s16::<1>(a1), vaddq_s16(a0, b1)),
            vdupq_n_s16(2),
        ));
        [
            vqmovun_s16(vbslq_s16(filtered, vbslq_s16(deep, strong0, weak0), a0)),
            vqmovun_s16(vbslq_s16(deep, strong1, a1)),
            vqmovun_s16(vbslq_s16(deep, strong2, a2)),
        ]
    };
    let [new_p0, new_p1, new_p2] = side([p0, p1, p2, p3], q0, q1);
    let [new_q0, new_q1, new_q2] = side([q0, q1, q2, q3], p0, p1);
    [new_p2, new_p1, new_p0, new_q0, new_q1, new_q2]
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
