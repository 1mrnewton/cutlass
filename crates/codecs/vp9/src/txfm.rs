#![allow(dead_code, unused_mut, unused_parens, clippy::double_parens)]
//! 8-bit inverse transforms.
//!
//! The arithmetic matches the libvpx scalar kernels, including the
//! 16-bit truncation on the DCT working arrays.

const COSPI_1: i32 = 16364;
const COSPI_2: i32 = 16305;
const COSPI_3: i32 = 16207;
const COSPI_4: i32 = 16069;
const COSPI_5: i32 = 15893;
const COSPI_6: i32 = 15679;
const COSPI_7: i32 = 15426;
const COSPI_8: i32 = 15137;
const COSPI_9: i32 = 14811;
const COSPI_10: i32 = 14449;
const COSPI_11: i32 = 14053;
const COSPI_12: i32 = 13623;
const COSPI_13: i32 = 13160;
const COSPI_14: i32 = 12665;
const COSPI_15: i32 = 12140;
const COSPI_16: i32 = 11585;
const COSPI_17: i32 = 11003;
const COSPI_18: i32 = 10394;
const COSPI_19: i32 = 9760;
const COSPI_20: i32 = 9102;
const COSPI_21: i32 = 8423;
const COSPI_22: i32 = 7723;
const COSPI_23: i32 = 7005;
const COSPI_24: i32 = 6270;
const COSPI_25: i32 = 5520;
const COSPI_26: i32 = 4756;
const COSPI_27: i32 = 3981;
const COSPI_28: i32 = 3196;
const COSPI_29: i32 = 2404;
const COSPI_30: i32 = 1606;
const COSPI_31: i32 = 804;
const SINPI_1: i32 = 5283;
const SINPI_2: i32 = 9929;
const SINPI_3: i32 = 13377;
const SINPI_4: i32 = 15212;

fn narrow(v: i32) -> i32 {
    i32::from(v as i16)
}

fn round_shift(v: i32) -> i32 {
    v.wrapping_add(8192) >> 14
}

pub(crate) fn round_pow2(v: i32, n: u32) -> i32 {
    if n == 0 {
        return v;
    }
    v.wrapping_add(1 << (n - 1)) >> n
}

fn clip_u8(v: i32) -> u8 {
    if v < 0 {
        0
    } else if v > 255 {
        255
    } else {
        v as u8
    }
}

fn iadst4(input: &[i32], output: &mut [i32]) {
    let mut s0: i32;
    let mut s1: i32;
    let mut s2: i32;
    let mut s3: i32;
    let mut s4: i32;
    let mut s5: i32;
    let mut s6: i32;
    let mut s7: i32;
    let mut x0: i32 = input[0];
    let mut x1: i32 = input[1];
    let mut x2: i32 = input[2];
    let mut x3: i32 = input[3];
    s0 = SINPI_1.wrapping_mul(x0);
    s1 = SINPI_2.wrapping_mul(x0);
    s2 = SINPI_3.wrapping_mul(x1);
    s3 = SINPI_4.wrapping_mul(x2);
    s4 = SINPI_1.wrapping_mul(x2);
    s5 = SINPI_2.wrapping_mul(x3);
    s6 = SINPI_4.wrapping_mul(x3);
    s7 = (x0.wrapping_sub(x2).wrapping_add(x3));
    s0 = s0.wrapping_add(s3).wrapping_add(s5);
    s1 = s1.wrapping_sub(s4).wrapping_sub(s6);
    s3 = s2;
    s2 = SINPI_3.wrapping_mul(s7);
    output[0] = (round_shift(s0.wrapping_add(s3)));
    output[1] = (round_shift(s1.wrapping_add(s3)));
    output[2] = (round_shift(s2));
    output[3] = (round_shift(s0.wrapping_add(s1).wrapping_sub(s3)));
}

fn idct4(input: &[i32], output: &mut [i32]) {
    let mut step = [0i32; 4];
    let mut temp1: i32;
    let mut temp2: i32;
    temp1 = (narrow(input[0]).wrapping_add(narrow(input[2]))).wrapping_mul(COSPI_16);
    temp2 = (narrow(input[0]).wrapping_sub(narrow(input[2]))).wrapping_mul(COSPI_16);
    step[0] = i32::from((round_shift(temp1)) as i16);
    step[1] = i32::from((round_shift(temp2)) as i16);
    temp1 = narrow(input[1])
        .wrapping_mul(COSPI_24)
        .wrapping_sub(narrow(input[3]).wrapping_mul(COSPI_8));
    temp2 = narrow(input[1])
        .wrapping_mul(COSPI_8)
        .wrapping_add(narrow(input[3]).wrapping_mul(COSPI_24));
    step[2] = i32::from((round_shift(temp1)) as i16);
    step[3] = i32::from((round_shift(temp2)) as i16);
    output[0] = (step[0].wrapping_add(step[3]));
    output[1] = (step[1].wrapping_add(step[2]));
    output[2] = (step[1].wrapping_sub(step[2]));
    output[3] = (step[0].wrapping_sub(step[3]));
}

fn iadst8(input: &[i32], output: &mut [i32]) {
    let mut s0: i32;
    let mut s1: i32;
    let mut s2: i32;
    let mut s3: i32;
    let mut s4: i32;
    let mut s5: i32;
    let mut s6: i32;
    let mut s7: i32;
    let mut x0: i32 = input[7];
    let mut x1: i32 = input[0];
    let mut x2: i32 = input[5];
    let mut x3: i32 = input[2];
    let mut x4: i32 = input[3];
    let mut x5: i32 = input[4];
    let mut x6: i32 = input[1];
    let mut x7: i32 = input[6];
    s0 = (COSPI_2
        .wrapping_mul(x0)
        .wrapping_add(COSPI_30.wrapping_mul(x1)));
    s1 = (COSPI_30
        .wrapping_mul(x0)
        .wrapping_sub(COSPI_2.wrapping_mul(x1)));
    s2 = (COSPI_10
        .wrapping_mul(x2)
        .wrapping_add(COSPI_22.wrapping_mul(x3)));
    s3 = (COSPI_22
        .wrapping_mul(x2)
        .wrapping_sub(COSPI_10.wrapping_mul(x3)));
    s4 = (COSPI_18
        .wrapping_mul(x4)
        .wrapping_add(COSPI_14.wrapping_mul(x5)));
    s5 = (COSPI_14
        .wrapping_mul(x4)
        .wrapping_sub(COSPI_18.wrapping_mul(x5)));
    s6 = (COSPI_26
        .wrapping_mul(x6)
        .wrapping_add(COSPI_6.wrapping_mul(x7)));
    s7 = (COSPI_6
        .wrapping_mul(x6)
        .wrapping_sub(COSPI_26.wrapping_mul(x7)));
    x0 = (round_shift(s0.wrapping_add(s4)));
    x1 = (round_shift(s1.wrapping_add(s5)));
    x2 = (round_shift(s2.wrapping_add(s6)));
    x3 = (round_shift(s3.wrapping_add(s7)));
    x4 = (round_shift(s0.wrapping_sub(s4)));
    x5 = (round_shift(s1.wrapping_sub(s5)));
    x6 = (round_shift(s2.wrapping_sub(s6)));
    x7 = (round_shift(s3.wrapping_sub(s7)));
    s0 = x0;
    s1 = x1;
    s2 = x2;
    s3 = x3;
    s4 = (COSPI_8
        .wrapping_mul(x4)
        .wrapping_add(COSPI_24.wrapping_mul(x5)));
    s5 = (COSPI_24
        .wrapping_mul(x4)
        .wrapping_sub(COSPI_8.wrapping_mul(x5)));
    s6 = ((COSPI_24)
        .wrapping_neg()
        .wrapping_mul(x6)
        .wrapping_add(COSPI_8.wrapping_mul(x7)));
    s7 = (COSPI_8
        .wrapping_mul(x6)
        .wrapping_add(COSPI_24.wrapping_mul(x7)));
    x0 = (s0.wrapping_add(s2));
    x1 = (s1.wrapping_add(s3));
    x2 = (s0.wrapping_sub(s2));
    x3 = (s1.wrapping_sub(s3));
    x4 = (round_shift(s4.wrapping_add(s6)));
    x5 = (round_shift(s5.wrapping_add(s7)));
    x6 = (round_shift(s4.wrapping_sub(s6)));
    x7 = (round_shift(s5.wrapping_sub(s7)));
    s2 = (COSPI_16.wrapping_mul((x2.wrapping_add(x3))));
    s3 = (COSPI_16.wrapping_mul((x2.wrapping_sub(x3))));
    s6 = (COSPI_16.wrapping_mul((x6.wrapping_add(x7))));
    s7 = (COSPI_16.wrapping_mul((x6.wrapping_sub(x7))));
    x2 = (round_shift(s2));
    x3 = (round_shift(s3));
    x6 = (round_shift(s6));
    x7 = (round_shift(s7));
    output[0] = (x0);
    output[1] = ((x4).wrapping_neg());
    output[2] = (x6);
    output[3] = ((x2).wrapping_neg());
    output[4] = (x3);
    output[5] = ((x7).wrapping_neg());
    output[6] = (x5);
    output[7] = ((x1).wrapping_neg());
}

fn idct8(input: &[i32], output: &mut [i32]) {
    let mut step1 = [0i32; 8];
    let mut step2 = [0i32; 8];
    let mut temp1: i32;
    let mut temp2: i32;
    step1[0] = i32::from((narrow(input[0])) as i16);
    step1[2] = i32::from((narrow(input[4])) as i16);
    step1[1] = i32::from((narrow(input[2])) as i16);
    step1[3] = i32::from((narrow(input[6])) as i16);
    temp1 = narrow(input[1])
        .wrapping_mul(COSPI_28)
        .wrapping_sub(narrow(input[7]).wrapping_mul(COSPI_4));
    temp2 = narrow(input[1])
        .wrapping_mul(COSPI_4)
        .wrapping_add(narrow(input[7]).wrapping_mul(COSPI_28));
    step1[4] = i32::from((round_shift(temp1)) as i16);
    step1[7] = i32::from((round_shift(temp2)) as i16);
    temp1 = narrow(input[5])
        .wrapping_mul(COSPI_12)
        .wrapping_sub(narrow(input[3]).wrapping_mul(COSPI_20));
    temp2 = narrow(input[5])
        .wrapping_mul(COSPI_20)
        .wrapping_add(narrow(input[3]).wrapping_mul(COSPI_12));
    step1[5] = i32::from((round_shift(temp1)) as i16);
    step1[6] = i32::from((round_shift(temp2)) as i16);
    temp1 = (step1[0].wrapping_add(step1[2])).wrapping_mul(COSPI_16);
    temp2 = (step1[0].wrapping_sub(step1[2])).wrapping_mul(COSPI_16);
    step2[0] = i32::from((round_shift(temp1)) as i16);
    step2[1] = i32::from((round_shift(temp2)) as i16);
    temp1 = step1[1]
        .wrapping_mul(COSPI_24)
        .wrapping_sub(step1[3].wrapping_mul(COSPI_8));
    temp2 = step1[1]
        .wrapping_mul(COSPI_8)
        .wrapping_add(step1[3].wrapping_mul(COSPI_24));
    step2[2] = i32::from((round_shift(temp1)) as i16);
    step2[3] = i32::from((round_shift(temp2)) as i16);
    step2[4] = i32::from((step1[4].wrapping_add(step1[5])) as i16);
    step2[5] = i32::from((step1[4].wrapping_sub(step1[5])) as i16);
    step2[6] = i32::from(((step1[6]).wrapping_neg().wrapping_add(step1[7])) as i16);
    step2[7] = i32::from((step1[6].wrapping_add(step1[7])) as i16);
    step1[0] = i32::from((step2[0].wrapping_add(step2[3])) as i16);
    step1[1] = i32::from((step2[1].wrapping_add(step2[2])) as i16);
    step1[2] = i32::from((step2[1].wrapping_sub(step2[2])) as i16);
    step1[3] = i32::from((step2[0].wrapping_sub(step2[3])) as i16);
    step1[4] = i32::from((step2[4]) as i16);
    temp1 = (step2[6].wrapping_sub(step2[5])).wrapping_mul(COSPI_16);
    temp2 = (step2[5].wrapping_add(step2[6])).wrapping_mul(COSPI_16);
    step1[5] = i32::from((round_shift(temp1)) as i16);
    step1[6] = i32::from((round_shift(temp2)) as i16);
    step1[7] = i32::from((step2[7]) as i16);
    output[0] = (step1[0].wrapping_add(step1[7]));
    output[1] = (step1[1].wrapping_add(step1[6]));
    output[2] = (step1[2].wrapping_add(step1[5]));
    output[3] = (step1[3].wrapping_add(step1[4]));
    output[4] = (step1[3].wrapping_sub(step1[4]));
    output[5] = (step1[2].wrapping_sub(step1[5]));
    output[6] = (step1[1].wrapping_sub(step1[6]));
    output[7] = (step1[0].wrapping_sub(step1[7]));
}

fn iadst16(input: &[i32], output: &mut [i32]) {
    let mut s0: i32;
    let mut s1: i32;
    let mut s2: i32;
    let mut s3: i32;
    let mut s4: i32;
    let mut s5: i32;
    let mut s6: i32;
    let mut s7: i32;
    let mut s8: i32;
    let mut s9: i32;
    let mut s10: i32;
    let mut s11: i32;
    let mut s12: i32;
    let mut s13: i32;
    let mut s14: i32;
    let mut s15: i32;
    let mut x0: i32 = input[15];
    let mut x1: i32 = input[0];
    let mut x2: i32 = input[13];
    let mut x3: i32 = input[2];
    let mut x4: i32 = input[11];
    let mut x5: i32 = input[4];
    let mut x6: i32 = input[9];
    let mut x7: i32 = input[6];
    let mut x8: i32 = input[7];
    let mut x9: i32 = input[8];
    let mut x10: i32 = input[5];
    let mut x11: i32 = input[10];
    let mut x12: i32 = input[3];
    let mut x13: i32 = input[12];
    let mut x14: i32 = input[1];
    let mut x15: i32 = input[14];
    s0 = x0
        .wrapping_mul(COSPI_1)
        .wrapping_add(x1.wrapping_mul(COSPI_31));
    s1 = x0
        .wrapping_mul(COSPI_31)
        .wrapping_sub(x1.wrapping_mul(COSPI_1));
    s2 = x2
        .wrapping_mul(COSPI_5)
        .wrapping_add(x3.wrapping_mul(COSPI_27));
    s3 = x2
        .wrapping_mul(COSPI_27)
        .wrapping_sub(x3.wrapping_mul(COSPI_5));
    s4 = x4
        .wrapping_mul(COSPI_9)
        .wrapping_add(x5.wrapping_mul(COSPI_23));
    s5 = x4
        .wrapping_mul(COSPI_23)
        .wrapping_sub(x5.wrapping_mul(COSPI_9));
    s6 = x6
        .wrapping_mul(COSPI_13)
        .wrapping_add(x7.wrapping_mul(COSPI_19));
    s7 = x6
        .wrapping_mul(COSPI_19)
        .wrapping_sub(x7.wrapping_mul(COSPI_13));
    s8 = x8
        .wrapping_mul(COSPI_17)
        .wrapping_add(x9.wrapping_mul(COSPI_15));
    s9 = x8
        .wrapping_mul(COSPI_15)
        .wrapping_sub(x9.wrapping_mul(COSPI_17));
    s10 = x10
        .wrapping_mul(COSPI_21)
        .wrapping_add(x11.wrapping_mul(COSPI_11));
    s11 = x10
        .wrapping_mul(COSPI_11)
        .wrapping_sub(x11.wrapping_mul(COSPI_21));
    s12 = x12
        .wrapping_mul(COSPI_25)
        .wrapping_add(x13.wrapping_mul(COSPI_7));
    s13 = x12
        .wrapping_mul(COSPI_7)
        .wrapping_sub(x13.wrapping_mul(COSPI_25));
    s14 = x14
        .wrapping_mul(COSPI_29)
        .wrapping_add(x15.wrapping_mul(COSPI_3));
    s15 = x14
        .wrapping_mul(COSPI_3)
        .wrapping_sub(x15.wrapping_mul(COSPI_29));
    x0 = (round_shift(s0.wrapping_add(s8)));
    x1 = (round_shift(s1.wrapping_add(s9)));
    x2 = (round_shift(s2.wrapping_add(s10)));
    x3 = (round_shift(s3.wrapping_add(s11)));
    x4 = (round_shift(s4.wrapping_add(s12)));
    x5 = (round_shift(s5.wrapping_add(s13)));
    x6 = (round_shift(s6.wrapping_add(s14)));
    x7 = (round_shift(s7.wrapping_add(s15)));
    x8 = (round_shift(s0.wrapping_sub(s8)));
    x9 = (round_shift(s1.wrapping_sub(s9)));
    x10 = (round_shift(s2.wrapping_sub(s10)));
    x11 = (round_shift(s3.wrapping_sub(s11)));
    x12 = (round_shift(s4.wrapping_sub(s12)));
    x13 = (round_shift(s5.wrapping_sub(s13)));
    x14 = (round_shift(s6.wrapping_sub(s14)));
    x15 = (round_shift(s7.wrapping_sub(s15)));
    s0 = x0;
    s1 = x1;
    s2 = x2;
    s3 = x3;
    s4 = x4;
    s5 = x5;
    s6 = x6;
    s7 = x7;
    s8 = x8
        .wrapping_mul(COSPI_4)
        .wrapping_add(x9.wrapping_mul(COSPI_28));
    s9 = x8
        .wrapping_mul(COSPI_28)
        .wrapping_sub(x9.wrapping_mul(COSPI_4));
    s10 = x10
        .wrapping_mul(COSPI_20)
        .wrapping_add(x11.wrapping_mul(COSPI_12));
    s11 = x10
        .wrapping_mul(COSPI_12)
        .wrapping_sub(x11.wrapping_mul(COSPI_20));
    s12 = (x12)
        .wrapping_neg()
        .wrapping_mul(COSPI_28)
        .wrapping_add(x13.wrapping_mul(COSPI_4));
    s13 = x12
        .wrapping_mul(COSPI_4)
        .wrapping_add(x13.wrapping_mul(COSPI_28));
    s14 = (x14)
        .wrapping_neg()
        .wrapping_mul(COSPI_12)
        .wrapping_add(x15.wrapping_mul(COSPI_20));
    s15 = x14
        .wrapping_mul(COSPI_20)
        .wrapping_add(x15.wrapping_mul(COSPI_12));
    x0 = (s0.wrapping_add(s4));
    x1 = (s1.wrapping_add(s5));
    x2 = (s2.wrapping_add(s6));
    x3 = (s3.wrapping_add(s7));
    x4 = (s0.wrapping_sub(s4));
    x5 = (s1.wrapping_sub(s5));
    x6 = (s2.wrapping_sub(s6));
    x7 = (s3.wrapping_sub(s7));
    x8 = (round_shift(s8.wrapping_add(s12)));
    x9 = (round_shift(s9.wrapping_add(s13)));
    x10 = (round_shift(s10.wrapping_add(s14)));
    x11 = (round_shift(s11.wrapping_add(s15)));
    x12 = (round_shift(s8.wrapping_sub(s12)));
    x13 = (round_shift(s9.wrapping_sub(s13)));
    x14 = (round_shift(s10.wrapping_sub(s14)));
    x15 = (round_shift(s11.wrapping_sub(s15)));
    s0 = x0;
    s1 = x1;
    s2 = x2;
    s3 = x3;
    s4 = x4
        .wrapping_mul(COSPI_8)
        .wrapping_add(x5.wrapping_mul(COSPI_24));
    s5 = x4
        .wrapping_mul(COSPI_24)
        .wrapping_sub(x5.wrapping_mul(COSPI_8));
    s6 = (x6)
        .wrapping_neg()
        .wrapping_mul(COSPI_24)
        .wrapping_add(x7.wrapping_mul(COSPI_8));
    s7 = x6
        .wrapping_mul(COSPI_8)
        .wrapping_add(x7.wrapping_mul(COSPI_24));
    s8 = x8;
    s9 = x9;
    s10 = x10;
    s11 = x11;
    s12 = x12
        .wrapping_mul(COSPI_8)
        .wrapping_add(x13.wrapping_mul(COSPI_24));
    s13 = x12
        .wrapping_mul(COSPI_24)
        .wrapping_sub(x13.wrapping_mul(COSPI_8));
    s14 = (x14)
        .wrapping_neg()
        .wrapping_mul(COSPI_24)
        .wrapping_add(x15.wrapping_mul(COSPI_8));
    s15 = x14
        .wrapping_mul(COSPI_8)
        .wrapping_add(x15.wrapping_mul(COSPI_24));
    x0 = (s0.wrapping_add(s2));
    x1 = (s1.wrapping_add(s3));
    x2 = (s0.wrapping_sub(s2));
    x3 = (s1.wrapping_sub(s3));
    x4 = (round_shift(s4.wrapping_add(s6)));
    x5 = (round_shift(s5.wrapping_add(s7)));
    x6 = (round_shift(s4.wrapping_sub(s6)));
    x7 = (round_shift(s5.wrapping_sub(s7)));
    x8 = (s8.wrapping_add(s10));
    x9 = (s9.wrapping_add(s11));
    x10 = (s8.wrapping_sub(s10));
    x11 = (s9.wrapping_sub(s11));
    x12 = (round_shift(s12.wrapping_add(s14)));
    x13 = (round_shift(s13.wrapping_add(s15)));
    x14 = (round_shift(s12.wrapping_sub(s14)));
    x15 = (round_shift(s13.wrapping_sub(s15)));
    s2 = ((COSPI_16).wrapping_neg()).wrapping_mul((x2.wrapping_add(x3)));
    s3 = COSPI_16.wrapping_mul((x2.wrapping_sub(x3)));
    s6 = COSPI_16.wrapping_mul((x6.wrapping_add(x7)));
    s7 = COSPI_16.wrapping_mul(((x6).wrapping_neg().wrapping_add(x7)));
    s10 = COSPI_16.wrapping_mul((x10.wrapping_add(x11)));
    s11 = COSPI_16.wrapping_mul(((x10).wrapping_neg().wrapping_add(x11)));
    s14 = ((COSPI_16).wrapping_neg()).wrapping_mul((x14.wrapping_add(x15)));
    s15 = COSPI_16.wrapping_mul((x14.wrapping_sub(x15)));
    x2 = (round_shift(s2));
    x3 = (round_shift(s3));
    x6 = (round_shift(s6));
    x7 = (round_shift(s7));
    x10 = (round_shift(s10));
    x11 = (round_shift(s11));
    x14 = (round_shift(s14));
    x15 = (round_shift(s15));
    output[0] = (x0);
    output[1] = ((x8).wrapping_neg());
    output[2] = (x12);
    output[3] = ((x4).wrapping_neg());
    output[4] = (x6);
    output[5] = (x14);
    output[6] = (x10);
    output[7] = (x2);
    output[8] = (x3);
    output[9] = (x11);
    output[10] = (x15);
    output[11] = (x7);
    output[12] = (x5);
    output[13] = ((x13).wrapping_neg());
    output[14] = (x9);
    output[15] = ((x1).wrapping_neg());
}

fn idct16(input: &[i32], output: &mut [i32]) {
    let mut step1 = [0i32; 16];
    let mut step2 = [0i32; 16];
    let mut temp1: i32;
    let mut temp2: i32;
    step1[0] = i32::from((narrow(input[0])) as i16);
    step1[1] = i32::from((narrow(input[8])) as i16);
    step1[2] = i32::from((narrow(input[4])) as i16);
    step1[3] = i32::from((narrow(input[12])) as i16);
    step1[4] = i32::from((narrow(input[2])) as i16);
    step1[5] = i32::from((narrow(input[10])) as i16);
    step1[6] = i32::from((narrow(input[6])) as i16);
    step1[7] = i32::from((narrow(input[14])) as i16);
    step1[8] = i32::from((narrow(input[1])) as i16);
    step1[9] = i32::from((narrow(input[9])) as i16);
    step1[10] = i32::from((narrow(input[5])) as i16);
    step1[11] = i32::from((narrow(input[13])) as i16);
    step1[12] = i32::from((narrow(input[3])) as i16);
    step1[13] = i32::from((narrow(input[11])) as i16);
    step1[14] = i32::from((narrow(input[7])) as i16);
    step1[15] = i32::from((narrow(input[15])) as i16);
    step2[0] = i32::from((step1[0]) as i16);
    step2[1] = i32::from((step1[1]) as i16);
    step2[2] = i32::from((step1[2]) as i16);
    step2[3] = i32::from((step1[3]) as i16);
    step2[4] = i32::from((step1[4]) as i16);
    step2[5] = i32::from((step1[5]) as i16);
    step2[6] = i32::from((step1[6]) as i16);
    step2[7] = i32::from((step1[7]) as i16);
    temp1 = step1[8]
        .wrapping_mul(COSPI_30)
        .wrapping_sub(step1[15].wrapping_mul(COSPI_2));
    temp2 = step1[8]
        .wrapping_mul(COSPI_2)
        .wrapping_add(step1[15].wrapping_mul(COSPI_30));
    step2[8] = i32::from((round_shift(temp1)) as i16);
    step2[15] = i32::from((round_shift(temp2)) as i16);
    temp1 = step1[9]
        .wrapping_mul(COSPI_14)
        .wrapping_sub(step1[14].wrapping_mul(COSPI_18));
    temp2 = step1[9]
        .wrapping_mul(COSPI_18)
        .wrapping_add(step1[14].wrapping_mul(COSPI_14));
    step2[9] = i32::from((round_shift(temp1)) as i16);
    step2[14] = i32::from((round_shift(temp2)) as i16);
    temp1 = step1[10]
        .wrapping_mul(COSPI_22)
        .wrapping_sub(step1[13].wrapping_mul(COSPI_10));
    temp2 = step1[10]
        .wrapping_mul(COSPI_10)
        .wrapping_add(step1[13].wrapping_mul(COSPI_22));
    step2[10] = i32::from((round_shift(temp1)) as i16);
    step2[13] = i32::from((round_shift(temp2)) as i16);
    temp1 = step1[11]
        .wrapping_mul(COSPI_6)
        .wrapping_sub(step1[12].wrapping_mul(COSPI_26));
    temp2 = step1[11]
        .wrapping_mul(COSPI_26)
        .wrapping_add(step1[12].wrapping_mul(COSPI_6));
    step2[11] = i32::from((round_shift(temp1)) as i16);
    step2[12] = i32::from((round_shift(temp2)) as i16);
    step1[0] = i32::from((step2[0]) as i16);
    step1[1] = i32::from((step2[1]) as i16);
    step1[2] = i32::from((step2[2]) as i16);
    step1[3] = i32::from((step2[3]) as i16);
    temp1 = step2[4]
        .wrapping_mul(COSPI_28)
        .wrapping_sub(step2[7].wrapping_mul(COSPI_4));
    temp2 = step2[4]
        .wrapping_mul(COSPI_4)
        .wrapping_add(step2[7].wrapping_mul(COSPI_28));
    step1[4] = i32::from((round_shift(temp1)) as i16);
    step1[7] = i32::from((round_shift(temp2)) as i16);
    temp1 = step2[5]
        .wrapping_mul(COSPI_12)
        .wrapping_sub(step2[6].wrapping_mul(COSPI_20));
    temp2 = step2[5]
        .wrapping_mul(COSPI_20)
        .wrapping_add(step2[6].wrapping_mul(COSPI_12));
    step1[5] = i32::from((round_shift(temp1)) as i16);
    step1[6] = i32::from((round_shift(temp2)) as i16);
    step1[8] = i32::from((step2[8].wrapping_add(step2[9])) as i16);
    step1[9] = i32::from((step2[8].wrapping_sub(step2[9])) as i16);
    step1[10] = i32::from(((step2[10]).wrapping_neg().wrapping_add(step2[11])) as i16);
    step1[11] = i32::from((step2[10].wrapping_add(step2[11])) as i16);
    step1[12] = i32::from((step2[12].wrapping_add(step2[13])) as i16);
    step1[13] = i32::from((step2[12].wrapping_sub(step2[13])) as i16);
    step1[14] = i32::from(((step2[14]).wrapping_neg().wrapping_add(step2[15])) as i16);
    step1[15] = i32::from((step2[14].wrapping_add(step2[15])) as i16);
    temp1 = (step1[0].wrapping_add(step1[1])).wrapping_mul(COSPI_16);
    temp2 = (step1[0].wrapping_sub(step1[1])).wrapping_mul(COSPI_16);
    step2[0] = i32::from((round_shift(temp1)) as i16);
    step2[1] = i32::from((round_shift(temp2)) as i16);
    temp1 = step1[2]
        .wrapping_mul(COSPI_24)
        .wrapping_sub(step1[3].wrapping_mul(COSPI_8));
    temp2 = step1[2]
        .wrapping_mul(COSPI_8)
        .wrapping_add(step1[3].wrapping_mul(COSPI_24));
    step2[2] = i32::from((round_shift(temp1)) as i16);
    step2[3] = i32::from((round_shift(temp2)) as i16);
    step2[4] = i32::from((step1[4].wrapping_add(step1[5])) as i16);
    step2[5] = i32::from((step1[4].wrapping_sub(step1[5])) as i16);
    step2[6] = i32::from(((step1[6]).wrapping_neg().wrapping_add(step1[7])) as i16);
    step2[7] = i32::from((step1[6].wrapping_add(step1[7])) as i16);
    step2[8] = i32::from((step1[8]) as i16);
    step2[15] = i32::from((step1[15]) as i16);
    temp1 = (step1[9])
        .wrapping_neg()
        .wrapping_mul(COSPI_8)
        .wrapping_add(step1[14].wrapping_mul(COSPI_24));
    temp2 = step1[9]
        .wrapping_mul(COSPI_24)
        .wrapping_add(step1[14].wrapping_mul(COSPI_8));
    step2[9] = i32::from((round_shift(temp1)) as i16);
    step2[14] = i32::from((round_shift(temp2)) as i16);
    temp1 = (step1[10])
        .wrapping_neg()
        .wrapping_mul(COSPI_24)
        .wrapping_sub(step1[13].wrapping_mul(COSPI_8));
    temp2 = (step1[10])
        .wrapping_neg()
        .wrapping_mul(COSPI_8)
        .wrapping_add(step1[13].wrapping_mul(COSPI_24));
    step2[10] = i32::from((round_shift(temp1)) as i16);
    step2[13] = i32::from((round_shift(temp2)) as i16);
    step2[11] = i32::from((step1[11]) as i16);
    step2[12] = i32::from((step1[12]) as i16);
    step1[0] = i32::from((step2[0].wrapping_add(step2[3])) as i16);
    step1[1] = i32::from((step2[1].wrapping_add(step2[2])) as i16);
    step1[2] = i32::from((step2[1].wrapping_sub(step2[2])) as i16);
    step1[3] = i32::from((step2[0].wrapping_sub(step2[3])) as i16);
    step1[4] = i32::from((step2[4]) as i16);
    temp1 = (step2[6].wrapping_sub(step2[5])).wrapping_mul(COSPI_16);
    temp2 = (step2[5].wrapping_add(step2[6])).wrapping_mul(COSPI_16);
    step1[5] = i32::from((round_shift(temp1)) as i16);
    step1[6] = i32::from((round_shift(temp2)) as i16);
    step1[7] = i32::from((step2[7]) as i16);
    step1[8] = i32::from((step2[8].wrapping_add(step2[11])) as i16);
    step1[9] = i32::from((step2[9].wrapping_add(step2[10])) as i16);
    step1[10] = i32::from((step2[9].wrapping_sub(step2[10])) as i16);
    step1[11] = i32::from((step2[8].wrapping_sub(step2[11])) as i16);
    step1[12] = i32::from(((step2[12]).wrapping_neg().wrapping_add(step2[15])) as i16);
    step1[13] = i32::from(((step2[13]).wrapping_neg().wrapping_add(step2[14])) as i16);
    step1[14] = i32::from((step2[13].wrapping_add(step2[14])) as i16);
    step1[15] = i32::from((step2[12].wrapping_add(step2[15])) as i16);
    step2[0] = i32::from((step1[0].wrapping_add(step1[7])) as i16);
    step2[1] = i32::from((step1[1].wrapping_add(step1[6])) as i16);
    step2[2] = i32::from((step1[2].wrapping_add(step1[5])) as i16);
    step2[3] = i32::from((step1[3].wrapping_add(step1[4])) as i16);
    step2[4] = i32::from((step1[3].wrapping_sub(step1[4])) as i16);
    step2[5] = i32::from((step1[2].wrapping_sub(step1[5])) as i16);
    step2[6] = i32::from((step1[1].wrapping_sub(step1[6])) as i16);
    step2[7] = i32::from((step1[0].wrapping_sub(step1[7])) as i16);
    step2[8] = i32::from((step1[8]) as i16);
    step2[9] = i32::from((step1[9]) as i16);
    temp1 = ((step1[10]).wrapping_neg().wrapping_add(step1[13])).wrapping_mul(COSPI_16);
    temp2 = (step1[10].wrapping_add(step1[13])).wrapping_mul(COSPI_16);
    step2[10] = i32::from((round_shift(temp1)) as i16);
    step2[13] = i32::from((round_shift(temp2)) as i16);
    temp1 = ((step1[11]).wrapping_neg().wrapping_add(step1[12])).wrapping_mul(COSPI_16);
    temp2 = (step1[11].wrapping_add(step1[12])).wrapping_mul(COSPI_16);
    step2[11] = i32::from((round_shift(temp1)) as i16);
    step2[12] = i32::from((round_shift(temp2)) as i16);
    step2[14] = i32::from((step1[14]) as i16);
    step2[15] = i32::from((step1[15]) as i16);
    output[0] = (step2[0].wrapping_add(step2[15]));
    output[1] = (step2[1].wrapping_add(step2[14]));
    output[2] = (step2[2].wrapping_add(step2[13]));
    output[3] = (step2[3].wrapping_add(step2[12]));
    output[4] = (step2[4].wrapping_add(step2[11]));
    output[5] = (step2[5].wrapping_add(step2[10]));
    output[6] = (step2[6].wrapping_add(step2[9]));
    output[7] = (step2[7].wrapping_add(step2[8]));
    output[8] = (step2[7].wrapping_sub(step2[8]));
    output[9] = (step2[6].wrapping_sub(step2[9]));
    output[10] = (step2[5].wrapping_sub(step2[10]));
    output[11] = (step2[4].wrapping_sub(step2[11]));
    output[12] = (step2[3].wrapping_sub(step2[12]));
    output[13] = (step2[2].wrapping_sub(step2[13]));
    output[14] = (step2[1].wrapping_sub(step2[14]));
    output[15] = (step2[0].wrapping_sub(step2[15]));
}

fn idct32(input: &[i32], output: &mut [i32]) {
    let mut step1 = [0i32; 32];
    let mut step2 = [0i32; 32];
    let mut temp1: i32;
    let mut temp2: i32;
    step1[0] = i32::from((narrow(input[0])) as i16);
    step1[1] = i32::from((narrow(input[16])) as i16);
    step1[2] = i32::from((narrow(input[8])) as i16);
    step1[3] = i32::from((narrow(input[24])) as i16);
    step1[4] = i32::from((narrow(input[4])) as i16);
    step1[5] = i32::from((narrow(input[20])) as i16);
    step1[6] = i32::from((narrow(input[12])) as i16);
    step1[7] = i32::from((narrow(input[28])) as i16);
    step1[8] = i32::from((narrow(input[2])) as i16);
    step1[9] = i32::from((narrow(input[18])) as i16);
    step1[10] = i32::from((narrow(input[10])) as i16);
    step1[11] = i32::from((narrow(input[26])) as i16);
    step1[12] = i32::from((narrow(input[6])) as i16);
    step1[13] = i32::from((narrow(input[22])) as i16);
    step1[14] = i32::from((narrow(input[14])) as i16);
    step1[15] = i32::from((narrow(input[30])) as i16);
    temp1 = narrow(input[1])
        .wrapping_mul(COSPI_31)
        .wrapping_sub(narrow(input[31]).wrapping_mul(COSPI_1));
    temp2 = narrow(input[1])
        .wrapping_mul(COSPI_1)
        .wrapping_add(narrow(input[31]).wrapping_mul(COSPI_31));
    step1[16] = i32::from((round_shift(temp1)) as i16);
    step1[31] = i32::from((round_shift(temp2)) as i16);
    temp1 = narrow(input[17])
        .wrapping_mul(COSPI_15)
        .wrapping_sub(narrow(input[15]).wrapping_mul(COSPI_17));
    temp2 = narrow(input[17])
        .wrapping_mul(COSPI_17)
        .wrapping_add(narrow(input[15]).wrapping_mul(COSPI_15));
    step1[17] = i32::from((round_shift(temp1)) as i16);
    step1[30] = i32::from((round_shift(temp2)) as i16);
    temp1 = narrow(input[9])
        .wrapping_mul(COSPI_23)
        .wrapping_sub(narrow(input[23]).wrapping_mul(COSPI_9));
    temp2 = narrow(input[9])
        .wrapping_mul(COSPI_9)
        .wrapping_add(narrow(input[23]).wrapping_mul(COSPI_23));
    step1[18] = i32::from((round_shift(temp1)) as i16);
    step1[29] = i32::from((round_shift(temp2)) as i16);
    temp1 = narrow(input[25])
        .wrapping_mul(COSPI_7)
        .wrapping_sub(narrow(input[7]).wrapping_mul(COSPI_25));
    temp2 = narrow(input[25])
        .wrapping_mul(COSPI_25)
        .wrapping_add(narrow(input[7]).wrapping_mul(COSPI_7));
    step1[19] = i32::from((round_shift(temp1)) as i16);
    step1[28] = i32::from((round_shift(temp2)) as i16);
    temp1 = narrow(input[5])
        .wrapping_mul(COSPI_27)
        .wrapping_sub(narrow(input[27]).wrapping_mul(COSPI_5));
    temp2 = narrow(input[5])
        .wrapping_mul(COSPI_5)
        .wrapping_add(narrow(input[27]).wrapping_mul(COSPI_27));
    step1[20] = i32::from((round_shift(temp1)) as i16);
    step1[27] = i32::from((round_shift(temp2)) as i16);
    temp1 = narrow(input[21])
        .wrapping_mul(COSPI_11)
        .wrapping_sub(narrow(input[11]).wrapping_mul(COSPI_21));
    temp2 = narrow(input[21])
        .wrapping_mul(COSPI_21)
        .wrapping_add(narrow(input[11]).wrapping_mul(COSPI_11));
    step1[21] = i32::from((round_shift(temp1)) as i16);
    step1[26] = i32::from((round_shift(temp2)) as i16);
    temp1 = narrow(input[13])
        .wrapping_mul(COSPI_19)
        .wrapping_sub(narrow(input[19]).wrapping_mul(COSPI_13));
    temp2 = narrow(input[13])
        .wrapping_mul(COSPI_13)
        .wrapping_add(narrow(input[19]).wrapping_mul(COSPI_19));
    step1[22] = i32::from((round_shift(temp1)) as i16);
    step1[25] = i32::from((round_shift(temp2)) as i16);
    temp1 = narrow(input[29])
        .wrapping_mul(COSPI_3)
        .wrapping_sub(narrow(input[3]).wrapping_mul(COSPI_29));
    temp2 = narrow(input[29])
        .wrapping_mul(COSPI_29)
        .wrapping_add(narrow(input[3]).wrapping_mul(COSPI_3));
    step1[23] = i32::from((round_shift(temp1)) as i16);
    step1[24] = i32::from((round_shift(temp2)) as i16);
    step2[0] = i32::from((step1[0]) as i16);
    step2[1] = i32::from((step1[1]) as i16);
    step2[2] = i32::from((step1[2]) as i16);
    step2[3] = i32::from((step1[3]) as i16);
    step2[4] = i32::from((step1[4]) as i16);
    step2[5] = i32::from((step1[5]) as i16);
    step2[6] = i32::from((step1[6]) as i16);
    step2[7] = i32::from((step1[7]) as i16);
    temp1 = step1[8]
        .wrapping_mul(COSPI_30)
        .wrapping_sub(step1[15].wrapping_mul(COSPI_2));
    temp2 = step1[8]
        .wrapping_mul(COSPI_2)
        .wrapping_add(step1[15].wrapping_mul(COSPI_30));
    step2[8] = i32::from((round_shift(temp1)) as i16);
    step2[15] = i32::from((round_shift(temp2)) as i16);
    temp1 = step1[9]
        .wrapping_mul(COSPI_14)
        .wrapping_sub(step1[14].wrapping_mul(COSPI_18));
    temp2 = step1[9]
        .wrapping_mul(COSPI_18)
        .wrapping_add(step1[14].wrapping_mul(COSPI_14));
    step2[9] = i32::from((round_shift(temp1)) as i16);
    step2[14] = i32::from((round_shift(temp2)) as i16);
    temp1 = step1[10]
        .wrapping_mul(COSPI_22)
        .wrapping_sub(step1[13].wrapping_mul(COSPI_10));
    temp2 = step1[10]
        .wrapping_mul(COSPI_10)
        .wrapping_add(step1[13].wrapping_mul(COSPI_22));
    step2[10] = i32::from((round_shift(temp1)) as i16);
    step2[13] = i32::from((round_shift(temp2)) as i16);
    temp1 = step1[11]
        .wrapping_mul(COSPI_6)
        .wrapping_sub(step1[12].wrapping_mul(COSPI_26));
    temp2 = step1[11]
        .wrapping_mul(COSPI_26)
        .wrapping_add(step1[12].wrapping_mul(COSPI_6));
    step2[11] = i32::from((round_shift(temp1)) as i16);
    step2[12] = i32::from((round_shift(temp2)) as i16);
    step2[16] = i32::from((step1[16].wrapping_add(step1[17])) as i16);
    step2[17] = i32::from((step1[16].wrapping_sub(step1[17])) as i16);
    step2[18] = i32::from(((step1[18]).wrapping_neg().wrapping_add(step1[19])) as i16);
    step2[19] = i32::from((step1[18].wrapping_add(step1[19])) as i16);
    step2[20] = i32::from((step1[20].wrapping_add(step1[21])) as i16);
    step2[21] = i32::from((step1[20].wrapping_sub(step1[21])) as i16);
    step2[22] = i32::from(((step1[22]).wrapping_neg().wrapping_add(step1[23])) as i16);
    step2[23] = i32::from((step1[22].wrapping_add(step1[23])) as i16);
    step2[24] = i32::from((step1[24].wrapping_add(step1[25])) as i16);
    step2[25] = i32::from((step1[24].wrapping_sub(step1[25])) as i16);
    step2[26] = i32::from(((step1[26]).wrapping_neg().wrapping_add(step1[27])) as i16);
    step2[27] = i32::from((step1[26].wrapping_add(step1[27])) as i16);
    step2[28] = i32::from((step1[28].wrapping_add(step1[29])) as i16);
    step2[29] = i32::from((step1[28].wrapping_sub(step1[29])) as i16);
    step2[30] = i32::from(((step1[30]).wrapping_neg().wrapping_add(step1[31])) as i16);
    step2[31] = i32::from((step1[30].wrapping_add(step1[31])) as i16);
    step1[0] = i32::from((step2[0]) as i16);
    step1[1] = i32::from((step2[1]) as i16);
    step1[2] = i32::from((step2[2]) as i16);
    step1[3] = i32::from((step2[3]) as i16);
    temp1 = step2[4]
        .wrapping_mul(COSPI_28)
        .wrapping_sub(step2[7].wrapping_mul(COSPI_4));
    temp2 = step2[4]
        .wrapping_mul(COSPI_4)
        .wrapping_add(step2[7].wrapping_mul(COSPI_28));
    step1[4] = i32::from((round_shift(temp1)) as i16);
    step1[7] = i32::from((round_shift(temp2)) as i16);
    temp1 = step2[5]
        .wrapping_mul(COSPI_12)
        .wrapping_sub(step2[6].wrapping_mul(COSPI_20));
    temp2 = step2[5]
        .wrapping_mul(COSPI_20)
        .wrapping_add(step2[6].wrapping_mul(COSPI_12));
    step1[5] = i32::from((round_shift(temp1)) as i16);
    step1[6] = i32::from((round_shift(temp2)) as i16);
    step1[8] = i32::from((step2[8].wrapping_add(step2[9])) as i16);
    step1[9] = i32::from((step2[8].wrapping_sub(step2[9])) as i16);
    step1[10] = i32::from(((step2[10]).wrapping_neg().wrapping_add(step2[11])) as i16);
    step1[11] = i32::from((step2[10].wrapping_add(step2[11])) as i16);
    step1[12] = i32::from((step2[12].wrapping_add(step2[13])) as i16);
    step1[13] = i32::from((step2[12].wrapping_sub(step2[13])) as i16);
    step1[14] = i32::from(((step2[14]).wrapping_neg().wrapping_add(step2[15])) as i16);
    step1[15] = i32::from((step2[14].wrapping_add(step2[15])) as i16);
    step1[16] = i32::from((step2[16]) as i16);
    step1[31] = i32::from((step2[31]) as i16);
    temp1 = (step2[17])
        .wrapping_neg()
        .wrapping_mul(COSPI_4)
        .wrapping_add(step2[30].wrapping_mul(COSPI_28));
    temp2 = step2[17]
        .wrapping_mul(COSPI_28)
        .wrapping_add(step2[30].wrapping_mul(COSPI_4));
    step1[17] = i32::from((round_shift(temp1)) as i16);
    step1[30] = i32::from((round_shift(temp2)) as i16);
    temp1 = (step2[18])
        .wrapping_neg()
        .wrapping_mul(COSPI_28)
        .wrapping_sub(step2[29].wrapping_mul(COSPI_4));
    temp2 = (step2[18])
        .wrapping_neg()
        .wrapping_mul(COSPI_4)
        .wrapping_add(step2[29].wrapping_mul(COSPI_28));
    step1[18] = i32::from((round_shift(temp1)) as i16);
    step1[29] = i32::from((round_shift(temp2)) as i16);
    step1[19] = i32::from((step2[19]) as i16);
    step1[20] = i32::from((step2[20]) as i16);
    temp1 = (step2[21])
        .wrapping_neg()
        .wrapping_mul(COSPI_20)
        .wrapping_add(step2[26].wrapping_mul(COSPI_12));
    temp2 = step2[21]
        .wrapping_mul(COSPI_12)
        .wrapping_add(step2[26].wrapping_mul(COSPI_20));
    step1[21] = i32::from((round_shift(temp1)) as i16);
    step1[26] = i32::from((round_shift(temp2)) as i16);
    temp1 = (step2[22])
        .wrapping_neg()
        .wrapping_mul(COSPI_12)
        .wrapping_sub(step2[25].wrapping_mul(COSPI_20));
    temp2 = (step2[22])
        .wrapping_neg()
        .wrapping_mul(COSPI_20)
        .wrapping_add(step2[25].wrapping_mul(COSPI_12));
    step1[22] = i32::from((round_shift(temp1)) as i16);
    step1[25] = i32::from((round_shift(temp2)) as i16);
    step1[23] = i32::from((step2[23]) as i16);
    step1[24] = i32::from((step2[24]) as i16);
    step1[27] = i32::from((step2[27]) as i16);
    step1[28] = i32::from((step2[28]) as i16);
    temp1 = (step1[0].wrapping_add(step1[1])).wrapping_mul(COSPI_16);
    temp2 = (step1[0].wrapping_sub(step1[1])).wrapping_mul(COSPI_16);
    step2[0] = i32::from((round_shift(temp1)) as i16);
    step2[1] = i32::from((round_shift(temp2)) as i16);
    temp1 = step1[2]
        .wrapping_mul(COSPI_24)
        .wrapping_sub(step1[3].wrapping_mul(COSPI_8));
    temp2 = step1[2]
        .wrapping_mul(COSPI_8)
        .wrapping_add(step1[3].wrapping_mul(COSPI_24));
    step2[2] = i32::from((round_shift(temp1)) as i16);
    step2[3] = i32::from((round_shift(temp2)) as i16);
    step2[4] = i32::from((step1[4].wrapping_add(step1[5])) as i16);
    step2[5] = i32::from((step1[4].wrapping_sub(step1[5])) as i16);
    step2[6] = i32::from(((step1[6]).wrapping_neg().wrapping_add(step1[7])) as i16);
    step2[7] = i32::from((step1[6].wrapping_add(step1[7])) as i16);
    step2[8] = i32::from((step1[8]) as i16);
    step2[15] = i32::from((step1[15]) as i16);
    temp1 = (step1[9])
        .wrapping_neg()
        .wrapping_mul(COSPI_8)
        .wrapping_add(step1[14].wrapping_mul(COSPI_24));
    temp2 = step1[9]
        .wrapping_mul(COSPI_24)
        .wrapping_add(step1[14].wrapping_mul(COSPI_8));
    step2[9] = i32::from((round_shift(temp1)) as i16);
    step2[14] = i32::from((round_shift(temp2)) as i16);
    temp1 = (step1[10])
        .wrapping_neg()
        .wrapping_mul(COSPI_24)
        .wrapping_sub(step1[13].wrapping_mul(COSPI_8));
    temp2 = (step1[10])
        .wrapping_neg()
        .wrapping_mul(COSPI_8)
        .wrapping_add(step1[13].wrapping_mul(COSPI_24));
    step2[10] = i32::from((round_shift(temp1)) as i16);
    step2[13] = i32::from((round_shift(temp2)) as i16);
    step2[11] = i32::from((step1[11]) as i16);
    step2[12] = i32::from((step1[12]) as i16);
    step2[16] = i32::from((step1[16].wrapping_add(step1[19])) as i16);
    step2[17] = i32::from((step1[17].wrapping_add(step1[18])) as i16);
    step2[18] = i32::from((step1[17].wrapping_sub(step1[18])) as i16);
    step2[19] = i32::from((step1[16].wrapping_sub(step1[19])) as i16);
    step2[20] = i32::from(((step1[20]).wrapping_neg().wrapping_add(step1[23])) as i16);
    step2[21] = i32::from(((step1[21]).wrapping_neg().wrapping_add(step1[22])) as i16);
    step2[22] = i32::from((step1[21].wrapping_add(step1[22])) as i16);
    step2[23] = i32::from((step1[20].wrapping_add(step1[23])) as i16);
    step2[24] = i32::from((step1[24].wrapping_add(step1[27])) as i16);
    step2[25] = i32::from((step1[25].wrapping_add(step1[26])) as i16);
    step2[26] = i32::from((step1[25].wrapping_sub(step1[26])) as i16);
    step2[27] = i32::from((step1[24].wrapping_sub(step1[27])) as i16);
    step2[28] = i32::from(((step1[28]).wrapping_neg().wrapping_add(step1[31])) as i16);
    step2[29] = i32::from(((step1[29]).wrapping_neg().wrapping_add(step1[30])) as i16);
    step2[30] = i32::from((step1[29].wrapping_add(step1[30])) as i16);
    step2[31] = i32::from((step1[28].wrapping_add(step1[31])) as i16);
    step1[0] = i32::from((step2[0].wrapping_add(step2[3])) as i16);
    step1[1] = i32::from((step2[1].wrapping_add(step2[2])) as i16);
    step1[2] = i32::from((step2[1].wrapping_sub(step2[2])) as i16);
    step1[3] = i32::from((step2[0].wrapping_sub(step2[3])) as i16);
    step1[4] = i32::from((step2[4]) as i16);
    temp1 = (step2[6].wrapping_sub(step2[5])).wrapping_mul(COSPI_16);
    temp2 = (step2[5].wrapping_add(step2[6])).wrapping_mul(COSPI_16);
    step1[5] = i32::from((round_shift(temp1)) as i16);
    step1[6] = i32::from((round_shift(temp2)) as i16);
    step1[7] = i32::from((step2[7]) as i16);
    step1[8] = i32::from((step2[8].wrapping_add(step2[11])) as i16);
    step1[9] = i32::from((step2[9].wrapping_add(step2[10])) as i16);
    step1[10] = i32::from((step2[9].wrapping_sub(step2[10])) as i16);
    step1[11] = i32::from((step2[8].wrapping_sub(step2[11])) as i16);
    step1[12] = i32::from(((step2[12]).wrapping_neg().wrapping_add(step2[15])) as i16);
    step1[13] = i32::from(((step2[13]).wrapping_neg().wrapping_add(step2[14])) as i16);
    step1[14] = i32::from((step2[13].wrapping_add(step2[14])) as i16);
    step1[15] = i32::from((step2[12].wrapping_add(step2[15])) as i16);
    step1[16] = i32::from((step2[16]) as i16);
    step1[17] = i32::from((step2[17]) as i16);
    temp1 = (step2[18])
        .wrapping_neg()
        .wrapping_mul(COSPI_8)
        .wrapping_add(step2[29].wrapping_mul(COSPI_24));
    temp2 = step2[18]
        .wrapping_mul(COSPI_24)
        .wrapping_add(step2[29].wrapping_mul(COSPI_8));
    step1[18] = i32::from((round_shift(temp1)) as i16);
    step1[29] = i32::from((round_shift(temp2)) as i16);
    temp1 = (step2[19])
        .wrapping_neg()
        .wrapping_mul(COSPI_8)
        .wrapping_add(step2[28].wrapping_mul(COSPI_24));
    temp2 = step2[19]
        .wrapping_mul(COSPI_24)
        .wrapping_add(step2[28].wrapping_mul(COSPI_8));
    step1[19] = i32::from((round_shift(temp1)) as i16);
    step1[28] = i32::from((round_shift(temp2)) as i16);
    temp1 = (step2[20])
        .wrapping_neg()
        .wrapping_mul(COSPI_24)
        .wrapping_sub(step2[27].wrapping_mul(COSPI_8));
    temp2 = (step2[20])
        .wrapping_neg()
        .wrapping_mul(COSPI_8)
        .wrapping_add(step2[27].wrapping_mul(COSPI_24));
    step1[20] = i32::from((round_shift(temp1)) as i16);
    step1[27] = i32::from((round_shift(temp2)) as i16);
    temp1 = (step2[21])
        .wrapping_neg()
        .wrapping_mul(COSPI_24)
        .wrapping_sub(step2[26].wrapping_mul(COSPI_8));
    temp2 = (step2[21])
        .wrapping_neg()
        .wrapping_mul(COSPI_8)
        .wrapping_add(step2[26].wrapping_mul(COSPI_24));
    step1[21] = i32::from((round_shift(temp1)) as i16);
    step1[26] = i32::from((round_shift(temp2)) as i16);
    step1[22] = i32::from((step2[22]) as i16);
    step1[23] = i32::from((step2[23]) as i16);
    step1[24] = i32::from((step2[24]) as i16);
    step1[25] = i32::from((step2[25]) as i16);
    step1[30] = i32::from((step2[30]) as i16);
    step1[31] = i32::from((step2[31]) as i16);
    step2[0] = i32::from((step1[0].wrapping_add(step1[7])) as i16);
    step2[1] = i32::from((step1[1].wrapping_add(step1[6])) as i16);
    step2[2] = i32::from((step1[2].wrapping_add(step1[5])) as i16);
    step2[3] = i32::from((step1[3].wrapping_add(step1[4])) as i16);
    step2[4] = i32::from((step1[3].wrapping_sub(step1[4])) as i16);
    step2[5] = i32::from((step1[2].wrapping_sub(step1[5])) as i16);
    step2[6] = i32::from((step1[1].wrapping_sub(step1[6])) as i16);
    step2[7] = i32::from((step1[0].wrapping_sub(step1[7])) as i16);
    step2[8] = i32::from((step1[8]) as i16);
    step2[9] = i32::from((step1[9]) as i16);
    temp1 = ((step1[10]).wrapping_neg().wrapping_add(step1[13])).wrapping_mul(COSPI_16);
    temp2 = (step1[10].wrapping_add(step1[13])).wrapping_mul(COSPI_16);
    step2[10] = i32::from((round_shift(temp1)) as i16);
    step2[13] = i32::from((round_shift(temp2)) as i16);
    temp1 = ((step1[11]).wrapping_neg().wrapping_add(step1[12])).wrapping_mul(COSPI_16);
    temp2 = (step1[11].wrapping_add(step1[12])).wrapping_mul(COSPI_16);
    step2[11] = i32::from((round_shift(temp1)) as i16);
    step2[12] = i32::from((round_shift(temp2)) as i16);
    step2[14] = i32::from((step1[14]) as i16);
    step2[15] = i32::from((step1[15]) as i16);
    step2[16] = i32::from((step1[16].wrapping_add(step1[23])) as i16);
    step2[17] = i32::from((step1[17].wrapping_add(step1[22])) as i16);
    step2[18] = i32::from((step1[18].wrapping_add(step1[21])) as i16);
    step2[19] = i32::from((step1[19].wrapping_add(step1[20])) as i16);
    step2[20] = i32::from((step1[19].wrapping_sub(step1[20])) as i16);
    step2[21] = i32::from((step1[18].wrapping_sub(step1[21])) as i16);
    step2[22] = i32::from((step1[17].wrapping_sub(step1[22])) as i16);
    step2[23] = i32::from((step1[16].wrapping_sub(step1[23])) as i16);
    step2[24] = i32::from(((step1[24]).wrapping_neg().wrapping_add(step1[31])) as i16);
    step2[25] = i32::from(((step1[25]).wrapping_neg().wrapping_add(step1[30])) as i16);
    step2[26] = i32::from(((step1[26]).wrapping_neg().wrapping_add(step1[29])) as i16);
    step2[27] = i32::from(((step1[27]).wrapping_neg().wrapping_add(step1[28])) as i16);
    step2[28] = i32::from((step1[27].wrapping_add(step1[28])) as i16);
    step2[29] = i32::from((step1[26].wrapping_add(step1[29])) as i16);
    step2[30] = i32::from((step1[25].wrapping_add(step1[30])) as i16);
    step2[31] = i32::from((step1[24].wrapping_add(step1[31])) as i16);
    step1[0] = i32::from((step2[0].wrapping_add(step2[15])) as i16);
    step1[1] = i32::from((step2[1].wrapping_add(step2[14])) as i16);
    step1[2] = i32::from((step2[2].wrapping_add(step2[13])) as i16);
    step1[3] = i32::from((step2[3].wrapping_add(step2[12])) as i16);
    step1[4] = i32::from((step2[4].wrapping_add(step2[11])) as i16);
    step1[5] = i32::from((step2[5].wrapping_add(step2[10])) as i16);
    step1[6] = i32::from((step2[6].wrapping_add(step2[9])) as i16);
    step1[7] = i32::from((step2[7].wrapping_add(step2[8])) as i16);
    step1[8] = i32::from((step2[7].wrapping_sub(step2[8])) as i16);
    step1[9] = i32::from((step2[6].wrapping_sub(step2[9])) as i16);
    step1[10] = i32::from((step2[5].wrapping_sub(step2[10])) as i16);
    step1[11] = i32::from((step2[4].wrapping_sub(step2[11])) as i16);
    step1[12] = i32::from((step2[3].wrapping_sub(step2[12])) as i16);
    step1[13] = i32::from((step2[2].wrapping_sub(step2[13])) as i16);
    step1[14] = i32::from((step2[1].wrapping_sub(step2[14])) as i16);
    step1[15] = i32::from((step2[0].wrapping_sub(step2[15])) as i16);
    step1[16] = i32::from((step2[16]) as i16);
    step1[17] = i32::from((step2[17]) as i16);
    step1[18] = i32::from((step2[18]) as i16);
    step1[19] = i32::from((step2[19]) as i16);
    temp1 = ((step2[20]).wrapping_neg().wrapping_add(step2[27])).wrapping_mul(COSPI_16);
    temp2 = (step2[20].wrapping_add(step2[27])).wrapping_mul(COSPI_16);
    step1[20] = i32::from((round_shift(temp1)) as i16);
    step1[27] = i32::from((round_shift(temp2)) as i16);
    temp1 = ((step2[21]).wrapping_neg().wrapping_add(step2[26])).wrapping_mul(COSPI_16);
    temp2 = (step2[21].wrapping_add(step2[26])).wrapping_mul(COSPI_16);
    step1[21] = i32::from((round_shift(temp1)) as i16);
    step1[26] = i32::from((round_shift(temp2)) as i16);
    temp1 = ((step2[22]).wrapping_neg().wrapping_add(step2[25])).wrapping_mul(COSPI_16);
    temp2 = (step2[22].wrapping_add(step2[25])).wrapping_mul(COSPI_16);
    step1[22] = i32::from((round_shift(temp1)) as i16);
    step1[25] = i32::from((round_shift(temp2)) as i16);
    temp1 = ((step2[23]).wrapping_neg().wrapping_add(step2[24])).wrapping_mul(COSPI_16);
    temp2 = (step2[23].wrapping_add(step2[24])).wrapping_mul(COSPI_16);
    step1[23] = i32::from((round_shift(temp1)) as i16);
    step1[24] = i32::from((round_shift(temp2)) as i16);
    step1[28] = i32::from((step2[28]) as i16);
    step1[29] = i32::from((step2[29]) as i16);
    step1[30] = i32::from((step2[30]) as i16);
    step1[31] = i32::from((step2[31]) as i16);
    output[0] = (step1[0].wrapping_add(step1[31]));
    output[1] = (step1[1].wrapping_add(step1[30]));
    output[2] = (step1[2].wrapping_add(step1[29]));
    output[3] = (step1[3].wrapping_add(step1[28]));
    output[4] = (step1[4].wrapping_add(step1[27]));
    output[5] = (step1[5].wrapping_add(step1[26]));
    output[6] = (step1[6].wrapping_add(step1[25]));
    output[7] = (step1[7].wrapping_add(step1[24]));
    output[8] = (step1[8].wrapping_add(step1[23]));
    output[9] = (step1[9].wrapping_add(step1[22]));
    output[10] = (step1[10].wrapping_add(step1[21]));
    output[11] = (step1[11].wrapping_add(step1[20]));
    output[12] = (step1[12].wrapping_add(step1[19]));
    output[13] = (step1[13].wrapping_add(step1[18]));
    output[14] = (step1[14].wrapping_add(step1[17]));
    output[15] = (step1[15].wrapping_add(step1[16]));
    output[16] = (step1[15].wrapping_sub(step1[16]));
    output[17] = (step1[14].wrapping_sub(step1[17]));
    output[18] = (step1[13].wrapping_sub(step1[18]));
    output[19] = (step1[12].wrapping_sub(step1[19]));
    output[20] = (step1[11].wrapping_sub(step1[20]));
    output[21] = (step1[10].wrapping_sub(step1[21]));
    output[22] = (step1[9].wrapping_sub(step1[22]));
    output[23] = (step1[8].wrapping_sub(step1[23]));
    output[24] = (step1[7].wrapping_sub(step1[24]));
    output[25] = (step1[6].wrapping_sub(step1[25]));
    output[26] = (step1[5].wrapping_sub(step1[26]));
    output[27] = (step1[4].wrapping_sub(step1[27]));
    output[28] = (step1[3].wrapping_sub(step1[28]));
    output[29] = (step1[2].wrapping_sub(step1[29]));
    output[30] = (step1[1].wrapping_sub(step1[30]));
    output[31] = (step1[0].wrapping_sub(step1[31]));
}

fn wht4(input: &[i32], output: &mut [i32], shift: u32) {
    let a0 = input[0] >> shift;
    let c0 = input[1] >> shift;
    let d0 = input[2] >> shift;
    let b0 = input[3] >> shift;
    let a1 = a0.wrapping_add(c0);
    let d1 = d0.wrapping_sub(b0);
    let e = a1.wrapping_sub(d1) >> 1;
    let b1 = e.wrapping_sub(b0);
    let c1 = e.wrapping_sub(c0);
    output[0] = a1.wrapping_sub(b1);
    output[1] = b1;
    output[2] = c1;
    output[3] = d1.wrapping_add(c1);
}

/// Inverse 4x4 Walsh-Hadamard added into `dest`.
pub(crate) fn iwht4x4_add(coef: &[i32], dest: &mut [u8], stride: usize) {
    let mut rows = [0i32; 16];
    let mut tmp = [0i32; 4];
    for r in 0..4 {
        wht4(&coef[r * 4..r * 4 + 4], &mut tmp, 2);
        rows[r * 4..r * 4 + 4].copy_from_slice(&tmp);
    }
    for c in 0..4 {
        let col = [rows[c], rows[4 + c], rows[8 + c], rows[12 + c]];
        wht4(&col, &mut tmp, 0);
        for (r, sample) in tmp.iter().enumerate() {
            let i = r * stride + c;
            let sum = i32::from(dest[i]).wrapping_add(*sample);
            dest[i] = clip_u8(sum);
        }
    }
}

fn transform_1d(n: usize, adst: bool, input: &[i32], output: &mut [i32]) {
    match (n, adst) {
        (4, false) => idct4(input, output),
        (4, true) => iadst4(input, output),
        (8, false) => idct8(input, output),
        (8, true) => iadst8(input, output),
        (16, false) => idct16(input, output),
        (16, true) => iadst16(input, output),
        _ => idct32(input, output),
    }
}

/// Row then column inverse transform, added into `dest` with the
/// libvpx column shift (4, 5, 6, 6).
pub(crate) fn iht_add(coef: &[i32], dest: &mut [u8], stride: usize, n: usize, tx_type: u8) {
    let row_adst = tx_type == 2 || tx_type == 3;
    let col_adst = tx_type == 1 || tx_type == 3;
    let shift = if n == 4 {
        4
    } else if n == 8 {
        5
    } else {
        6
    };
    let mut rows = vec![0i32; n * n];
    let mut tmp_in = vec![0i32; n];
    let mut tmp_out = vec![0i32; n];
    for r in 0..n {
        transform_1d(n, row_adst, &coef[r * n..r * n + n], &mut tmp_out);
        rows[r * n..r * n + n].copy_from_slice(&tmp_out);
    }
    for c in 0..n {
        for r in 0..n {
            tmp_in[r] = rows[r * n + c];
        }
        transform_1d(n, col_adst && n <= 16, &tmp_in, &mut tmp_out);
        for (r, sample) in tmp_out.iter().enumerate() {
            let add = round_pow2(*sample, shift);
            let i = r * stride + c;
            dest[i] = clip_u8(i32::from(dest[i]).wrapping_add(add));
        }
    }
}

#[cfg(test)]
mod oracle {
    use super::{iht_add, iwht4x4_add};

    #[test]
    fn zero_residual_does_not_change_pixels() {
        let mut dest = [9u8; 16];
        iwht4x4_add(&[0; 16], &mut dest, 4);
        assert_eq!(dest, [9u8; 16]);
        iht_add(&[0; 16], &mut dest, 4, 4, 0);
        assert_eq!(dest, [9u8; 16]);
        let mut block = [4u8; 64];
        iht_add(&[0; 64], &mut block, 8, 8, 3);
        assert_eq!(block, [4u8; 64]);
    }

    use super::{iadst4, iadst8, iadst16, idct4, idct8, idct16, idct32};

    fn check(f: fn(&[i32], &mut [i32]), input: &[i32], expect: &[i32]) {
        let mut out = vec![0i32; expect.len()];
        f(input, &mut out);
        assert_eq!(out, expect);
    }
    #[test]
    fn idct4_case_0() {
        check(idct4, &[-163, 282, -236, -39], &[-36, 196, -92, -528]);
    }
    #[test]
    fn idct4_case_1() {
        check(idct4, &[-180, 207, 160, 183], &[247, -330, -150, -275]);
    }
    #[test]
    fn idct4_case_2() {
        check(idct4, &[88, -86, -204, 199], &[-85, -11, 423, -79]);
    }
    #[test]
    fn idct8_case_3() {
        check(
            idct8,
            &[-271, 99, 143, -298, 156, -28, -66, -196],
            &[-178, 91, -240, 166, -542, -596, -463, 230],
        );
    }
    #[test]
    fn idct8_case_4() {
        check(
            idct8,
            &[25, -269, -278, -274, 254, -291, 90, -79],
            &[-693, -193, 24, 354, 484, 32, -511, 643],
        );
    }
    #[test]
    fn idct8_case_5() {
        check(
            idct8,
            &[132, -271, 240, -73, 148, 207, 266, -62],
            &[298, -545, 53, 95, -347, 233, 215, 746],
        );
    }
    #[test]
    fn idct16_case_6() {
        check(
            idct16,
            &[
                53, -64, -76, 170, -4, -278, 126, 269, -198, -110, 3, -177, 40, 212, 132, 219,
            ],
            &[
                -43, -86, 312, -651, 705, -230, -356, -449, 153, 960, 772, -511, -3, 6, 38, -25,
            ],
        );
    }
    #[test]
    fn idct16_case_7() {
        check(
            idct16,
            &[
                -106, 10, -10, 211, 217, 102, -265, 191, -52, 113, 124, -123, 75, 261, 83, -212,
            ],
            &[
                490, -201, 112, -365, -461, -474, -455, 786, -262, 655, -322, -555, 17, 476, -95,
                -546,
            ],
        );
    }
    #[test]
    fn idct16_case_8() {
        check(
            idct16,
            &[
                149, 220, -190, -133, 233, 102, 79, 201, -270, 180, -256, 15, 292, 103, -126, -128,
            ],
            &[
                429, 88, 47, -254, 101, 1135, -44, 862, 196, -22, 495, -587, -912, 231, 438, -523,
            ],
        );
    }
    #[test]
    fn idct32_case_9() {
        check(
            idct32,
            &[
                214, -68, -288, -96, 252, 261, -63, 114, 226, 52, 291, 61, 170, -25, 261, -295, 92,
                224, -168, 231, 274, -90, 136, -243, 192, 73, 283, 267, -96, 216, 123, 196,
            ],
            &[
                1511, -840, 144, -1330, -120, -690, 490, 37, -693, 191, 608, -304, 1255, 273, 1071,
                1017, 697, 145, 493, 969, -704, -890, 85, 925, -365, -626, 1896, -240, -54, -418,
                -556, 855,
            ],
        );
    }
    #[test]
    fn idct32_case_10() {
        check(
            idct32,
            &[
                65, 124, 54, -299, 251, 253, 39, 169, -272, -65, -119, 263, 298, -115, -207, 264,
                -39, -267, -228, -215, -283, 163, -286, -13, -45, -25, -188, -111, 52, -3, -229,
                -129,
            ],
            &[
                -203, 1633, -402, -383, 221, 728, -828, -720, -179, 401, 253, 1668, 893, 342,
                -1200, 610, 644, 300, -262, -1359, -6, 289, -805, -1403, -90, -320, -62, 877, 385,
                210, 907, -667,
            ],
        );
    }
    #[test]
    fn idct32_case_11() {
        check(
            idct32,
            &[
                -137, -39, 240, -128, -21, 1, 165, 29, 208, 185, -184, -276, 19, 95, 51, 131, -108,
                -36, -189, -41, 222, -86, 142, -279, -70, -282, 106, -151, -264, -136, 156, 218,
            ],
            &[
                -60, 823, -632, 258, -532, -515, 59, -565, 1033, 797, -91, -1565, 415, -956, 599,
                -299, -119, -915, -1076, -131, -297, 153, 191, -247, -521, -187, 617, -674, -178,
                1180, -383, 714,
            ],
        );
    }
    #[test]
    fn iadst4_case_12() {
        check(iadst4, &[136, 257, -75, 228], &[322, 105, 358, -203]);
    }
    #[test]
    fn iadst4_case_13() {
        check(iadst4, &[161, -72, 236, -269], &[49, 212, -281, 438]);
    }
    #[test]
    fn iadst4_case_14() {
        check(iadst4, &[104, 289, 28, 136], &[378, 164, 173, -166]);
    }
    #[test]
    fn iadst8_case_15() {
        check(
            iadst8,
            &[-240, 5, -172, -83, -252, 13, -228, -222],
            &[-778, -73, -180, 86, -205, -216, 58, -542],
        );
    }
    #[test]
    fn iadst8_case_16() {
        check(
            iadst8,
            &[17, 5, -138, 126, 278, -42, -167, -292],
            &[-254, 503, -619, 10, 386, 161, -183, -30],
        );
    }
    #[test]
    fn iadst8_case_17() {
        check(
            iadst8,
            &[274, -262, -78, 283, 171, -125, 221, -262],
            &[66, 253, -540, -146, -19, 972, 33, 493],
        );
    }
    #[test]
    fn iadst16_case_18() {
        check(
            iadst16,
            &[
                87, -95, 55, -199, -90, 287, 143, -102, 204, -194, 99, 3, 216, 211, -283, 33,
            ],
            &[
                307, 105, -302, -99, -688, 503, 352, 177, 368, -789, -121, -467, 1055, -32, -161,
                565,
            ],
        );
    }
    #[test]
    fn iadst16_case_19() {
        check(
            iadst16,
            &[
                111, -12, -282, -140, -95, 35, 276, -162, 47, 139, -82, -28, -202, 88, 260, 52,
            ],
            &[
                176, -396, -163, -891, 71, -6, 275, 880, -274, 159, 89, 875, 49, -476, 49, 72,
            ],
        );
    }
    #[test]
    fn iadst16_case_20() {
        check(
            iadst16,
            &[
                247, 196, 245, -60, -234, -259, -214, -164, -127, -130, 251, -82, -26, 40, 218, -39,
            ],
            &[
                -257, -847, 359, 1005, 855, 379, 313, 337, -573, 444, -72, -87, 213, 470, -215, 479,
            ],
        );
    }
}
