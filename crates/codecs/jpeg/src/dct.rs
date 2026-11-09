//! Integer 8×8 DCT, the Loeffler–Ligtenberg–Moschytz factorisation with
//! 13-bit constants and two extra bits between passes.
//!
//! This is the accurate integer transform of the Independent JPEG Group, so
//! a decode matches libjpeg's `-dct int` sample for sample. Pass 1 runs on
//! coefficients clamped to the 16-bit range, which keeps every 32-bit sum in
//! range. Pass 2 runs in 64 bits because its inputs are not bounded that way.

const CONST_BITS: u32 = 13;
const PASS1_BITS: u32 = 2;

const FIX_0_298631336: i32 = 2446;
const FIX_0_390180644: i32 = 3196;
const FIX_0_541196100: i32 = 4433;
const FIX_0_765366865: i32 = 6270;
const FIX_0_899976223: i32 = 7373;
const FIX_1_175875602: i32 = 9633;
const FIX_1_501321110: i32 = 12299;
const FIX_1_847759065: i32 = 15137;
const FIX_1_961570560: i32 = 16069;
const FIX_2_053119869: i32 = 16819;
const FIX_2_562915447: i32 = 20995;
const FIX_3_072711026: i32 = 25172;

/// Coefficient bound for pass 1. libjpeg stores coefficients in 16 bits.
const COEFF_LIMIT: i32 = 32_767;

fn descale32(value: i32, shift: u32) -> i32 {
    (value + (1 << (shift - 1))) >> shift
}

fn clamp_sample(value: i64) -> u8 {
    u8::try_from(value.clamp(0, 255)).unwrap_or(0)
}

/// Dequantize and inverse-transform one block of natural-order coefficients
/// into `out`, eight samples per row, rows `stride` apart.
pub(crate) fn idct_block(coeffs: &[i32; 64], quant: &[u16; 64], out: &mut [u8], stride: usize) {
    let mut work = [0i32; 64];
    for col in 0..8 {
        let x = |row: usize| -> i32 {
            let value = i64::from(coeffs[row * 8 + col]) * i64::from(quant[row * 8 + col]);
            i32::try_from(value.clamp(-i64::from(COEFF_LIMIT), i64::from(COEFF_LIMIT))).unwrap_or(0)
        };
        if (1..8).all(|row| coeffs[row * 8 + col] == 0) {
            let dc = x(0) << PASS1_BITS;
            for row in 0..8 {
                work[row * 8 + col] = dc;
            }
            continue;
        }
        let (z2, z3) = (x(2), x(6));
        let z1 = (z2 + z3) * FIX_0_541196100;
        let tmp2 = z1 - z3 * FIX_1_847759065;
        let tmp3 = z1 + z2 * FIX_0_765366865;
        let (z2, z3) = (x(0), x(4));
        let tmp0 = (z2 + z3) << CONST_BITS;
        let tmp1 = (z2 - z3) << CONST_BITS;
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        let (mut t0, mut t1, mut t2, mut t3) = (x(7), x(5), x(3), x(1));
        let z1 = t0 + t3;
        let z2 = t1 + t2;
        let z3 = t0 + t2;
        let z4 = t1 + t3;
        let z5 = (z3 + z4) * FIX_1_175875602;
        t0 *= FIX_0_298631336;
        t1 *= FIX_2_053119869;
        t2 *= FIX_3_072711026;
        t3 *= FIX_1_501321110;
        let z1 = -z1 * FIX_0_899976223;
        let z2 = -z2 * FIX_2_562915447;
        let z3 = -z3 * FIX_1_961570560 + z5;
        let z4 = -z4 * FIX_0_390180644 + z5;
        t0 += z1 + z3;
        t1 += z2 + z4;
        t2 += z2 + z3;
        t3 += z1 + z4;

        let shift = CONST_BITS - PASS1_BITS;
        work[col] = descale32(tmp10 + t3, shift);
        work[7 * 8 + col] = descale32(tmp10 - t3, shift);
        work[8 + col] = descale32(tmp11 + t2, shift);
        work[6 * 8 + col] = descale32(tmp11 - t2, shift);
        work[2 * 8 + col] = descale32(tmp12 + t1, shift);
        work[5 * 8 + col] = descale32(tmp12 - t1, shift);
        work[3 * 8 + col] = descale32(tmp13 + t0, shift);
        work[4 * 8 + col] = descale32(tmp13 - t0, shift);
    }

    let shift = CONST_BITS + PASS1_BITS + 3;
    let round = 1i64 << (shift - 1);
    for row in 0..8 {
        let w = &work[row * 8..row * 8 + 8];
        let Some(dest) = out.get_mut(row * stride..row * stride + 8) else {
            return;
        };
        if w[1..].iter().all(|value| *value == 0) {
            let level = clamp_sample(((i64::from(w[0]) + 16) >> (PASS1_BITS + 3)) + 128);
            dest.fill(level);
            continue;
        }
        let x = |index: usize| i64::from(w[index]);
        let (z2, z3) = (x(2), x(6));
        let z1 = (z2 + z3) * i64::from(FIX_0_541196100);
        let tmp2 = z1 - z3 * i64::from(FIX_1_847759065);
        let tmp3 = z1 + z2 * i64::from(FIX_0_765366865);
        let tmp0 = (x(0) + x(4)) << CONST_BITS;
        let tmp1 = (x(0) - x(4)) << CONST_BITS;
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        let (mut t0, mut t1, mut t2, mut t3) = (x(7), x(5), x(3), x(1));
        let z1 = t0 + t3;
        let z2 = t1 + t2;
        let z3 = t0 + t2;
        let z4 = t1 + t3;
        let z5 = (z3 + z4) * i64::from(FIX_1_175875602);
        t0 *= i64::from(FIX_0_298631336);
        t1 *= i64::from(FIX_2_053119869);
        t2 *= i64::from(FIX_3_072711026);
        t3 *= i64::from(FIX_1_501321110);
        let z1 = -z1 * i64::from(FIX_0_899976223);
        let z2 = -z2 * i64::from(FIX_2_562915447);
        let z3 = -z3 * i64::from(FIX_1_961570560) + z5;
        let z4 = -z4 * i64::from(FIX_0_390180644) + z5;
        t0 += z1 + z3;
        t1 += z2 + z4;
        t2 += z2 + z3;
        t3 += z1 + z4;

        let put = |value: i64| clamp_sample(((value + round) >> shift) + 128);
        dest[0] = put(tmp10 + t3);
        dest[7] = put(tmp10 - t3);
        dest[1] = put(tmp11 + t2);
        dest[6] = put(tmp11 - t2);
        dest[2] = put(tmp12 + t1);
        dest[5] = put(tmp12 - t1);
        dest[3] = put(tmp13 + t0);
        dest[4] = put(tmp13 - t0);
    }
}

/// A block whose only coefficient is DC. Equal to [`idct_block`] on that block.
pub(crate) fn idct_dc(dc: i32, quant: u16, out: &mut [u8], stride: usize) {
    let value =
        (i64::from(dc) * i64::from(quant)).clamp(-i64::from(COEFF_LIMIT), i64::from(COEFF_LIMIT));
    let level = clamp_sample((((value << PASS1_BITS) + 16) >> (PASS1_BITS + 3)) + 128);
    for row in 0..8 {
        if let Some(dest) = out.get_mut(row * stride..row * stride + 8) {
            dest.fill(level);
        }
    }
}

/// Forward transform of one block of level-shifted samples, in place.
/// The result is the T.81 coefficient scaled by 8.
pub(crate) fn fdct_block(data: &mut [i32; 64]) {
    for row in 0..8 {
        let d = &mut data[row * 8..row * 8 + 8];
        let tmp0 = d[0] + d[7];
        let tmp7 = d[0] - d[7];
        let tmp1 = d[1] + d[6];
        let tmp6 = d[1] - d[6];
        let tmp2 = d[2] + d[5];
        let tmp5 = d[2] - d[5];
        let tmp3 = d[3] + d[4];
        let tmp4 = d[3] - d[4];

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        d[0] = (tmp10 + tmp11) << PASS1_BITS;
        d[4] = (tmp10 - tmp11) << PASS1_BITS;
        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        let shift = CONST_BITS - PASS1_BITS;
        d[2] = descale32(z1 + tmp13 * FIX_0_765366865, shift);
        d[6] = descale32(z1 - tmp12 * FIX_1_847759065, shift);

        let (o7, o5, o3, o1) = odd(tmp4, tmp5, tmp6, tmp7);
        d[7] = descale32(o7, shift);
        d[5] = descale32(o5, shift);
        d[3] = descale32(o3, shift);
        d[1] = descale32(o1, shift);
    }
    for col in 0..8 {
        let at = |row: usize| row * 8 + col;
        let tmp0 = data[at(0)] + data[at(7)];
        let tmp7 = data[at(0)] - data[at(7)];
        let tmp1 = data[at(1)] + data[at(6)];
        let tmp6 = data[at(1)] - data[at(6)];
        let tmp2 = data[at(2)] + data[at(5)];
        let tmp5 = data[at(2)] - data[at(5)];
        let tmp3 = data[at(3)] + data[at(4)];
        let tmp4 = data[at(3)] - data[at(4)];

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        data[at(0)] = descale32(tmp10 + tmp11, PASS1_BITS);
        data[at(4)] = descale32(tmp10 - tmp11, PASS1_BITS);
        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        let shift = CONST_BITS + PASS1_BITS;
        data[at(2)] = descale32(z1 + tmp13 * FIX_0_765366865, shift);
        data[at(6)] = descale32(z1 - tmp12 * FIX_1_847759065, shift);

        let (o7, o5, o3, o1) = odd(tmp4, tmp5, tmp6, tmp7);
        data[at(7)] = descale32(o7, shift);
        data[at(5)] = descale32(o5, shift);
        data[at(3)] = descale32(o3, shift);
        data[at(1)] = descale32(o1, shift);
    }
}

/// Odd half of the forward butterfly: outputs 7, 5, 3, 1.
fn odd(tmp4: i32, tmp5: i32, tmp6: i32, tmp7: i32) -> (i32, i32, i32, i32) {
    let z1 = tmp4 + tmp7;
    let z2 = tmp5 + tmp6;
    let z3 = tmp4 + tmp6;
    let z4 = tmp5 + tmp7;
    let z5 = (z3 + z4) * FIX_1_175875602;
    let tmp4 = tmp4 * FIX_0_298631336;
    let tmp5 = tmp5 * FIX_2_053119869;
    let tmp6 = tmp6 * FIX_3_072711026;
    let tmp7 = tmp7 * FIX_1_501321110;
    let z1 = -z1 * FIX_0_899976223;
    let z2 = -z2 * FIX_2_562915447;
    let z3 = -z3 * FIX_1_961570560 + z5;
    let z4 = -z4 * FIX_0_390180644 + z5;
    (
        tmp4 + z1 + z3,
        tmp5 + z2 + z4,
        tmp6 + z2 + z3,
        tmp7 + z1 + z4,
    )
}
