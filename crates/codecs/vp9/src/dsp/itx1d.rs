//! The 1-D inverse transforms of VP9 (libvpx `vpx_dsp/inv_txfm.c`), on
//! lanes: each value is a vector of independent transforms, so one call
//! runs a transform over several rows or columns at once and the compiler
//! keeps the lanes in vector registers. Generated from the scalar
//! versions in `txfm.rs`, which the tests there check against libvpx.
#![allow(unused_mut, unused_parens, clippy::double_parens, clippy::needless_range_loop)]

use super::lanes::{K, Lanes, rs};

const COSPI_1: K = K(16364);
const COSPI_2: K = K(16305);
const COSPI_3: K = K(16207);
const COSPI_4: K = K(16069);
const COSPI_5: K = K(15893);
const COSPI_6: K = K(15679);
const COSPI_7: K = K(15426);
const COSPI_8: K = K(15137);
const COSPI_9: K = K(14811);
const COSPI_10: K = K(14449);
const COSPI_11: K = K(14053);
const COSPI_12: K = K(13623);
const COSPI_13: K = K(13160);
const COSPI_14: K = K(12665);
const COSPI_15: K = K(12140);
const COSPI_16: K = K(11585);
const COSPI_17: K = K(11003);
const COSPI_18: K = K(10394);
const COSPI_19: K = K(9760);
const COSPI_20: K = K(9102);
const COSPI_21: K = K(8423);
const COSPI_22: K = K(7723);
const COSPI_23: K = K(7005);
const COSPI_24: K = K(6270);
const COSPI_25: K = K(5520);
const COSPI_26: K = K(4756);
const COSPI_27: K = K(3981);
const COSPI_28: K = K(3196);
const COSPI_29: K = K(2404);
const COSPI_30: K = K(1606);
const COSPI_31: K = K(804);
const SINPI_1: K = K(5283);
const SINPI_2: K = K(9929);
const SINPI_3: K = K(13377);
const SINPI_4: K = K(15212);






#[inline(always)]
pub fn iadst4<V: Lanes>(input: &[V], output: &mut [V]) {
    let mut s0: V;
    let mut s1: V;
    let mut s2: V;
    let mut s3: V;
    let mut s4: V;
    let mut s5: V;
    let mut s6: V;
    let mut s7: V;
    let mut x0: V = input[0];
    let mut x1: V = input[1];
    let mut x2: V = input[2];
    let mut x3: V = input[3];
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
    output[0] = (rs(s0.wrapping_add(s3)));
    output[1] = (rs(s1.wrapping_add(s3)));
    output[2] = (rs(s2));
    output[3] = (rs(s0.wrapping_add(s1).wrapping_sub(s3)));
}

#[inline(always)]
pub fn idct4<V: Lanes>(input: &[V], output: &mut [V]) {
    let mut step = [V::ZERO; 4];
    let mut temp1: V;
    let mut temp2: V;
    temp1 = ((input[0]).wrapping_add((input[2]))).wrapping_mul(COSPI_16);
    temp2 = ((input[0]).wrapping_sub((input[2]))).wrapping_mul(COSPI_16);
    step[0] = (rs(temp1));
    step[1] = (rs(temp2));
    temp1 = (input[1])
        .wrapping_mul(COSPI_24)
        .wrapping_sub((input[3]).wrapping_mul(COSPI_8));
    temp2 = (input[1])
        .wrapping_mul(COSPI_8)
        .wrapping_add((input[3]).wrapping_mul(COSPI_24));
    step[2] = (rs(temp1));
    step[3] = (rs(temp2));
    output[0] = (step[0].wrapping_add(step[3]));
    output[1] = (step[1].wrapping_add(step[2]));
    output[2] = (step[1].wrapping_sub(step[2]));
    output[3] = (step[0].wrapping_sub(step[3]));
}

#[inline(always)]
pub fn iadst8<V: Lanes>(input: &[V], output: &mut [V]) {
    let mut s0: V;
    let mut s1: V;
    let mut s2: V;
    let mut s3: V;
    let mut s4: V;
    let mut s5: V;
    let mut s6: V;
    let mut s7: V;
    let mut x0: V = input[7];
    let mut x1: V = input[0];
    let mut x2: V = input[5];
    let mut x3: V = input[2];
    let mut x4: V = input[3];
    let mut x5: V = input[4];
    let mut x6: V = input[1];
    let mut x7: V = input[6];
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
    x0 = (rs(s0.wrapping_add(s4)));
    x1 = (rs(s1.wrapping_add(s5)));
    x2 = (rs(s2.wrapping_add(s6)));
    x3 = (rs(s3.wrapping_add(s7)));
    x4 = (rs(s0.wrapping_sub(s4)));
    x5 = (rs(s1.wrapping_sub(s5)));
    x6 = (rs(s2.wrapping_sub(s6)));
    x7 = (rs(s3.wrapping_sub(s7)));
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
    x4 = (rs(s4.wrapping_add(s6)));
    x5 = (rs(s5.wrapping_add(s7)));
    x6 = (rs(s4.wrapping_sub(s6)));
    x7 = (rs(s5.wrapping_sub(s7)));
    s2 = (COSPI_16.wrapping_mul((x2.wrapping_add(x3))));
    s3 = (COSPI_16.wrapping_mul((x2.wrapping_sub(x3))));
    s6 = (COSPI_16.wrapping_mul((x6.wrapping_add(x7))));
    s7 = (COSPI_16.wrapping_mul((x6.wrapping_sub(x7))));
    x2 = (rs(s2));
    x3 = (rs(s3));
    x6 = (rs(s6));
    x7 = (rs(s7));
    output[0] = (x0);
    output[1] = ((x4).wrapping_neg());
    output[2] = (x6);
    output[3] = ((x2).wrapping_neg());
    output[4] = (x3);
    output[5] = ((x7).wrapping_neg());
    output[6] = (x5);
    output[7] = ((x1).wrapping_neg());
}

#[inline(always)]
pub fn idct8<V: Lanes>(input: &[V], output: &mut [V]) {
    let mut step1 = [V::ZERO; 8];
    let mut step2 = [V::ZERO; 8];
    let mut temp1: V;
    let mut temp2: V;
    step1[0] = ((input[0]));
    step1[2] = ((input[4]));
    step1[1] = ((input[2]));
    step1[3] = ((input[6]));
    temp1 = (input[1])
        .wrapping_mul(COSPI_28)
        .wrapping_sub((input[7]).wrapping_mul(COSPI_4));
    temp2 = (input[1])
        .wrapping_mul(COSPI_4)
        .wrapping_add((input[7]).wrapping_mul(COSPI_28));
    step1[4] = (rs(temp1));
    step1[7] = (rs(temp2));
    temp1 = (input[5])
        .wrapping_mul(COSPI_12)
        .wrapping_sub((input[3]).wrapping_mul(COSPI_20));
    temp2 = (input[5])
        .wrapping_mul(COSPI_20)
        .wrapping_add((input[3]).wrapping_mul(COSPI_12));
    step1[5] = (rs(temp1));
    step1[6] = (rs(temp2));
    temp1 = (step1[0].wrapping_add(step1[2])).wrapping_mul(COSPI_16);
    temp2 = (step1[0].wrapping_sub(step1[2])).wrapping_mul(COSPI_16);
    step2[0] = (rs(temp1));
    step2[1] = (rs(temp2));
    temp1 = step1[1]
        .wrapping_mul(COSPI_24)
        .wrapping_sub(step1[3].wrapping_mul(COSPI_8));
    temp2 = step1[1]
        .wrapping_mul(COSPI_8)
        .wrapping_add(step1[3].wrapping_mul(COSPI_24));
    step2[2] = (rs(temp1));
    step2[3] = (rs(temp2));
    step2[4] = (step1[4].wrapping_add(step1[5]));
    step2[5] = (step1[4].wrapping_sub(step1[5]));
    step2[6] = ((step1[6]).wrapping_neg().wrapping_add(step1[7]));
    step2[7] = (step1[6].wrapping_add(step1[7]));
    step1[0] = (step2[0].wrapping_add(step2[3]));
    step1[1] = (step2[1].wrapping_add(step2[2]));
    step1[2] = (step2[1].wrapping_sub(step2[2]));
    step1[3] = (step2[0].wrapping_sub(step2[3]));
    step1[4] = (step2[4]);
    temp1 = (step2[6].wrapping_sub(step2[5])).wrapping_mul(COSPI_16);
    temp2 = (step2[5].wrapping_add(step2[6])).wrapping_mul(COSPI_16);
    step1[5] = (rs(temp1));
    step1[6] = (rs(temp2));
    step1[7] = (step2[7]);
    output[0] = (step1[0].wrapping_add(step1[7]));
    output[1] = (step1[1].wrapping_add(step1[6]));
    output[2] = (step1[2].wrapping_add(step1[5]));
    output[3] = (step1[3].wrapping_add(step1[4]));
    output[4] = (step1[3].wrapping_sub(step1[4]));
    output[5] = (step1[2].wrapping_sub(step1[5]));
    output[6] = (step1[1].wrapping_sub(step1[6]));
    output[7] = (step1[0].wrapping_sub(step1[7]));
}

#[inline(always)]
pub fn iadst16<V: Lanes>(input: &[V], output: &mut [V]) {
    let mut s0: V;
    let mut s1: V;
    let mut s2: V;
    let mut s3: V;
    let mut s4: V;
    let mut s5: V;
    let mut s6: V;
    let mut s7: V;
    let mut s8: V;
    let mut s9: V;
    let mut s10: V;
    let mut s11: V;
    let mut s12: V;
    let mut s13: V;
    let mut s14: V;
    let mut s15: V;
    let mut x0: V = input[15];
    let mut x1: V = input[0];
    let mut x2: V = input[13];
    let mut x3: V = input[2];
    let mut x4: V = input[11];
    let mut x5: V = input[4];
    let mut x6: V = input[9];
    let mut x7: V = input[6];
    let mut x8: V = input[7];
    let mut x9: V = input[8];
    let mut x10: V = input[5];
    let mut x11: V = input[10];
    let mut x12: V = input[3];
    let mut x13: V = input[12];
    let mut x14: V = input[1];
    let mut x15: V = input[14];
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
    x0 = (rs(s0.wrapping_add(s8)));
    x1 = (rs(s1.wrapping_add(s9)));
    x2 = (rs(s2.wrapping_add(s10)));
    x3 = (rs(s3.wrapping_add(s11)));
    x4 = (rs(s4.wrapping_add(s12)));
    x5 = (rs(s5.wrapping_add(s13)));
    x6 = (rs(s6.wrapping_add(s14)));
    x7 = (rs(s7.wrapping_add(s15)));
    x8 = (rs(s0.wrapping_sub(s8)));
    x9 = (rs(s1.wrapping_sub(s9)));
    x10 = (rs(s2.wrapping_sub(s10)));
    x11 = (rs(s3.wrapping_sub(s11)));
    x12 = (rs(s4.wrapping_sub(s12)));
    x13 = (rs(s5.wrapping_sub(s13)));
    x14 = (rs(s6.wrapping_sub(s14)));
    x15 = (rs(s7.wrapping_sub(s15)));
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
    x8 = (rs(s8.wrapping_add(s12)));
    x9 = (rs(s9.wrapping_add(s13)));
    x10 = (rs(s10.wrapping_add(s14)));
    x11 = (rs(s11.wrapping_add(s15)));
    x12 = (rs(s8.wrapping_sub(s12)));
    x13 = (rs(s9.wrapping_sub(s13)));
    x14 = (rs(s10.wrapping_sub(s14)));
    x15 = (rs(s11.wrapping_sub(s15)));
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
    x4 = (rs(s4.wrapping_add(s6)));
    x5 = (rs(s5.wrapping_add(s7)));
    x6 = (rs(s4.wrapping_sub(s6)));
    x7 = (rs(s5.wrapping_sub(s7)));
    x8 = (s8.wrapping_add(s10));
    x9 = (s9.wrapping_add(s11));
    x10 = (s8.wrapping_sub(s10));
    x11 = (s9.wrapping_sub(s11));
    x12 = (rs(s12.wrapping_add(s14)));
    x13 = (rs(s13.wrapping_add(s15)));
    x14 = (rs(s12.wrapping_sub(s14)));
    x15 = (rs(s13.wrapping_sub(s15)));
    s2 = ((COSPI_16).wrapping_neg()).wrapping_mul((x2.wrapping_add(x3)));
    s3 = COSPI_16.wrapping_mul((x2.wrapping_sub(x3)));
    s6 = COSPI_16.wrapping_mul((x6.wrapping_add(x7)));
    s7 = COSPI_16.wrapping_mul(((x6).wrapping_neg().wrapping_add(x7)));
    s10 = COSPI_16.wrapping_mul((x10.wrapping_add(x11)));
    s11 = COSPI_16.wrapping_mul(((x10).wrapping_neg().wrapping_add(x11)));
    s14 = ((COSPI_16).wrapping_neg()).wrapping_mul((x14.wrapping_add(x15)));
    s15 = COSPI_16.wrapping_mul((x14.wrapping_sub(x15)));
    x2 = (rs(s2));
    x3 = (rs(s3));
    x6 = (rs(s6));
    x7 = (rs(s7));
    x10 = (rs(s10));
    x11 = (rs(s11));
    x14 = (rs(s14));
    x15 = (rs(s15));
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

#[inline(always)]
pub fn idct16<V: Lanes>(input: &[V], output: &mut [V]) {
    let mut step1 = [V::ZERO; 16];
    let mut step2 = [V::ZERO; 16];
    let mut temp1: V;
    let mut temp2: V;
    step1[0] = ((input[0]));
    step1[1] = ((input[8]));
    step1[2] = ((input[4]));
    step1[3] = ((input[12]));
    step1[4] = ((input[2]));
    step1[5] = ((input[10]));
    step1[6] = ((input[6]));
    step1[7] = ((input[14]));
    step1[8] = ((input[1]));
    step1[9] = ((input[9]));
    step1[10] = ((input[5]));
    step1[11] = ((input[13]));
    step1[12] = ((input[3]));
    step1[13] = ((input[11]));
    step1[14] = ((input[7]));
    step1[15] = ((input[15]));
    step2[0] = (step1[0]);
    step2[1] = (step1[1]);
    step2[2] = (step1[2]);
    step2[3] = (step1[3]);
    step2[4] = (step1[4]);
    step2[5] = (step1[5]);
    step2[6] = (step1[6]);
    step2[7] = (step1[7]);
    temp1 = step1[8]
        .wrapping_mul(COSPI_30)
        .wrapping_sub(step1[15].wrapping_mul(COSPI_2));
    temp2 = step1[8]
        .wrapping_mul(COSPI_2)
        .wrapping_add(step1[15].wrapping_mul(COSPI_30));
    step2[8] = (rs(temp1));
    step2[15] = (rs(temp2));
    temp1 = step1[9]
        .wrapping_mul(COSPI_14)
        .wrapping_sub(step1[14].wrapping_mul(COSPI_18));
    temp2 = step1[9]
        .wrapping_mul(COSPI_18)
        .wrapping_add(step1[14].wrapping_mul(COSPI_14));
    step2[9] = (rs(temp1));
    step2[14] = (rs(temp2));
    temp1 = step1[10]
        .wrapping_mul(COSPI_22)
        .wrapping_sub(step1[13].wrapping_mul(COSPI_10));
    temp2 = step1[10]
        .wrapping_mul(COSPI_10)
        .wrapping_add(step1[13].wrapping_mul(COSPI_22));
    step2[10] = (rs(temp1));
    step2[13] = (rs(temp2));
    temp1 = step1[11]
        .wrapping_mul(COSPI_6)
        .wrapping_sub(step1[12].wrapping_mul(COSPI_26));
    temp2 = step1[11]
        .wrapping_mul(COSPI_26)
        .wrapping_add(step1[12].wrapping_mul(COSPI_6));
    step2[11] = (rs(temp1));
    step2[12] = (rs(temp2));
    step1[0] = (step2[0]);
    step1[1] = (step2[1]);
    step1[2] = (step2[2]);
    step1[3] = (step2[3]);
    temp1 = step2[4]
        .wrapping_mul(COSPI_28)
        .wrapping_sub(step2[7].wrapping_mul(COSPI_4));
    temp2 = step2[4]
        .wrapping_mul(COSPI_4)
        .wrapping_add(step2[7].wrapping_mul(COSPI_28));
    step1[4] = (rs(temp1));
    step1[7] = (rs(temp2));
    temp1 = step2[5]
        .wrapping_mul(COSPI_12)
        .wrapping_sub(step2[6].wrapping_mul(COSPI_20));
    temp2 = step2[5]
        .wrapping_mul(COSPI_20)
        .wrapping_add(step2[6].wrapping_mul(COSPI_12));
    step1[5] = (rs(temp1));
    step1[6] = (rs(temp2));
    step1[8] = (step2[8].wrapping_add(step2[9]));
    step1[9] = (step2[8].wrapping_sub(step2[9]));
    step1[10] = ((step2[10]).wrapping_neg().wrapping_add(step2[11]));
    step1[11] = (step2[10].wrapping_add(step2[11]));
    step1[12] = (step2[12].wrapping_add(step2[13]));
    step1[13] = (step2[12].wrapping_sub(step2[13]));
    step1[14] = ((step2[14]).wrapping_neg().wrapping_add(step2[15]));
    step1[15] = (step2[14].wrapping_add(step2[15]));
    temp1 = (step1[0].wrapping_add(step1[1])).wrapping_mul(COSPI_16);
    temp2 = (step1[0].wrapping_sub(step1[1])).wrapping_mul(COSPI_16);
    step2[0] = (rs(temp1));
    step2[1] = (rs(temp2));
    temp1 = step1[2]
        .wrapping_mul(COSPI_24)
        .wrapping_sub(step1[3].wrapping_mul(COSPI_8));
    temp2 = step1[2]
        .wrapping_mul(COSPI_8)
        .wrapping_add(step1[3].wrapping_mul(COSPI_24));
    step2[2] = (rs(temp1));
    step2[3] = (rs(temp2));
    step2[4] = (step1[4].wrapping_add(step1[5]));
    step2[5] = (step1[4].wrapping_sub(step1[5]));
    step2[6] = ((step1[6]).wrapping_neg().wrapping_add(step1[7]));
    step2[7] = (step1[6].wrapping_add(step1[7]));
    step2[8] = (step1[8]);
    step2[15] = (step1[15]);
    temp1 = (step1[9])
        .wrapping_neg()
        .wrapping_mul(COSPI_8)
        .wrapping_add(step1[14].wrapping_mul(COSPI_24));
    temp2 = step1[9]
        .wrapping_mul(COSPI_24)
        .wrapping_add(step1[14].wrapping_mul(COSPI_8));
    step2[9] = (rs(temp1));
    step2[14] = (rs(temp2));
    temp1 = (step1[10])
        .wrapping_neg()
        .wrapping_mul(COSPI_24)
        .wrapping_sub(step1[13].wrapping_mul(COSPI_8));
    temp2 = (step1[10])
        .wrapping_neg()
        .wrapping_mul(COSPI_8)
        .wrapping_add(step1[13].wrapping_mul(COSPI_24));
    step2[10] = (rs(temp1));
    step2[13] = (rs(temp2));
    step2[11] = (step1[11]);
    step2[12] = (step1[12]);
    step1[0] = (step2[0].wrapping_add(step2[3]));
    step1[1] = (step2[1].wrapping_add(step2[2]));
    step1[2] = (step2[1].wrapping_sub(step2[2]));
    step1[3] = (step2[0].wrapping_sub(step2[3]));
    step1[4] = (step2[4]);
    temp1 = (step2[6].wrapping_sub(step2[5])).wrapping_mul(COSPI_16);
    temp2 = (step2[5].wrapping_add(step2[6])).wrapping_mul(COSPI_16);
    step1[5] = (rs(temp1));
    step1[6] = (rs(temp2));
    step1[7] = (step2[7]);
    step1[8] = (step2[8].wrapping_add(step2[11]));
    step1[9] = (step2[9].wrapping_add(step2[10]));
    step1[10] = (step2[9].wrapping_sub(step2[10]));
    step1[11] = (step2[8].wrapping_sub(step2[11]));
    step1[12] = ((step2[12]).wrapping_neg().wrapping_add(step2[15]));
    step1[13] = ((step2[13]).wrapping_neg().wrapping_add(step2[14]));
    step1[14] = (step2[13].wrapping_add(step2[14]));
    step1[15] = (step2[12].wrapping_add(step2[15]));
    step2[0] = (step1[0].wrapping_add(step1[7]));
    step2[1] = (step1[1].wrapping_add(step1[6]));
    step2[2] = (step1[2].wrapping_add(step1[5]));
    step2[3] = (step1[3].wrapping_add(step1[4]));
    step2[4] = (step1[3].wrapping_sub(step1[4]));
    step2[5] = (step1[2].wrapping_sub(step1[5]));
    step2[6] = (step1[1].wrapping_sub(step1[6]));
    step2[7] = (step1[0].wrapping_sub(step1[7]));
    step2[8] = (step1[8]);
    step2[9] = (step1[9]);
    temp1 = ((step1[10]).wrapping_neg().wrapping_add(step1[13])).wrapping_mul(COSPI_16);
    temp2 = (step1[10].wrapping_add(step1[13])).wrapping_mul(COSPI_16);
    step2[10] = (rs(temp1));
    step2[13] = (rs(temp2));
    temp1 = ((step1[11]).wrapping_neg().wrapping_add(step1[12])).wrapping_mul(COSPI_16);
    temp2 = (step1[11].wrapping_add(step1[12])).wrapping_mul(COSPI_16);
    step2[11] = (rs(temp1));
    step2[12] = (rs(temp2));
    step2[14] = (step1[14]);
    step2[15] = (step1[15]);
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

#[inline(always)]
pub fn idct32<V: Lanes>(input: &[V], output: &mut [V]) {
    let mut step1 = [V::ZERO; 32];
    let mut step2 = [V::ZERO; 32];
    let mut temp1: V;
    let mut temp2: V;
    step1[0] = ((input[0]));
    step1[1] = ((input[16]));
    step1[2] = ((input[8]));
    step1[3] = ((input[24]));
    step1[4] = ((input[4]));
    step1[5] = ((input[20]));
    step1[6] = ((input[12]));
    step1[7] = ((input[28]));
    step1[8] = ((input[2]));
    step1[9] = ((input[18]));
    step1[10] = ((input[10]));
    step1[11] = ((input[26]));
    step1[12] = ((input[6]));
    step1[13] = ((input[22]));
    step1[14] = ((input[14]));
    step1[15] = ((input[30]));
    temp1 = (input[1])
        .wrapping_mul(COSPI_31)
        .wrapping_sub((input[31]).wrapping_mul(COSPI_1));
    temp2 = (input[1])
        .wrapping_mul(COSPI_1)
        .wrapping_add((input[31]).wrapping_mul(COSPI_31));
    step1[16] = (rs(temp1));
    step1[31] = (rs(temp2));
    temp1 = (input[17])
        .wrapping_mul(COSPI_15)
        .wrapping_sub((input[15]).wrapping_mul(COSPI_17));
    temp2 = (input[17])
        .wrapping_mul(COSPI_17)
        .wrapping_add((input[15]).wrapping_mul(COSPI_15));
    step1[17] = (rs(temp1));
    step1[30] = (rs(temp2));
    temp1 = (input[9])
        .wrapping_mul(COSPI_23)
        .wrapping_sub((input[23]).wrapping_mul(COSPI_9));
    temp2 = (input[9])
        .wrapping_mul(COSPI_9)
        .wrapping_add((input[23]).wrapping_mul(COSPI_23));
    step1[18] = (rs(temp1));
    step1[29] = (rs(temp2));
    temp1 = (input[25])
        .wrapping_mul(COSPI_7)
        .wrapping_sub((input[7]).wrapping_mul(COSPI_25));
    temp2 = (input[25])
        .wrapping_mul(COSPI_25)
        .wrapping_add((input[7]).wrapping_mul(COSPI_7));
    step1[19] = (rs(temp1));
    step1[28] = (rs(temp2));
    temp1 = (input[5])
        .wrapping_mul(COSPI_27)
        .wrapping_sub((input[27]).wrapping_mul(COSPI_5));
    temp2 = (input[5])
        .wrapping_mul(COSPI_5)
        .wrapping_add((input[27]).wrapping_mul(COSPI_27));
    step1[20] = (rs(temp1));
    step1[27] = (rs(temp2));
    temp1 = (input[21])
        .wrapping_mul(COSPI_11)
        .wrapping_sub((input[11]).wrapping_mul(COSPI_21));
    temp2 = (input[21])
        .wrapping_mul(COSPI_21)
        .wrapping_add((input[11]).wrapping_mul(COSPI_11));
    step1[21] = (rs(temp1));
    step1[26] = (rs(temp2));
    temp1 = (input[13])
        .wrapping_mul(COSPI_19)
        .wrapping_sub((input[19]).wrapping_mul(COSPI_13));
    temp2 = (input[13])
        .wrapping_mul(COSPI_13)
        .wrapping_add((input[19]).wrapping_mul(COSPI_19));
    step1[22] = (rs(temp1));
    step1[25] = (rs(temp2));
    temp1 = (input[29])
        .wrapping_mul(COSPI_3)
        .wrapping_sub((input[3]).wrapping_mul(COSPI_29));
    temp2 = (input[29])
        .wrapping_mul(COSPI_29)
        .wrapping_add((input[3]).wrapping_mul(COSPI_3));
    step1[23] = (rs(temp1));
    step1[24] = (rs(temp2));
    step2[0] = (step1[0]);
    step2[1] = (step1[1]);
    step2[2] = (step1[2]);
    step2[3] = (step1[3]);
    step2[4] = (step1[4]);
    step2[5] = (step1[5]);
    step2[6] = (step1[6]);
    step2[7] = (step1[7]);
    temp1 = step1[8]
        .wrapping_mul(COSPI_30)
        .wrapping_sub(step1[15].wrapping_mul(COSPI_2));
    temp2 = step1[8]
        .wrapping_mul(COSPI_2)
        .wrapping_add(step1[15].wrapping_mul(COSPI_30));
    step2[8] = (rs(temp1));
    step2[15] = (rs(temp2));
    temp1 = step1[9]
        .wrapping_mul(COSPI_14)
        .wrapping_sub(step1[14].wrapping_mul(COSPI_18));
    temp2 = step1[9]
        .wrapping_mul(COSPI_18)
        .wrapping_add(step1[14].wrapping_mul(COSPI_14));
    step2[9] = (rs(temp1));
    step2[14] = (rs(temp2));
    temp1 = step1[10]
        .wrapping_mul(COSPI_22)
        .wrapping_sub(step1[13].wrapping_mul(COSPI_10));
    temp2 = step1[10]
        .wrapping_mul(COSPI_10)
        .wrapping_add(step1[13].wrapping_mul(COSPI_22));
    step2[10] = (rs(temp1));
    step2[13] = (rs(temp2));
    temp1 = step1[11]
        .wrapping_mul(COSPI_6)
        .wrapping_sub(step1[12].wrapping_mul(COSPI_26));
    temp2 = step1[11]
        .wrapping_mul(COSPI_26)
        .wrapping_add(step1[12].wrapping_mul(COSPI_6));
    step2[11] = (rs(temp1));
    step2[12] = (rs(temp2));
    step2[16] = (step1[16].wrapping_add(step1[17]));
    step2[17] = (step1[16].wrapping_sub(step1[17]));
    step2[18] = ((step1[18]).wrapping_neg().wrapping_add(step1[19]));
    step2[19] = (step1[18].wrapping_add(step1[19]));
    step2[20] = (step1[20].wrapping_add(step1[21]));
    step2[21] = (step1[20].wrapping_sub(step1[21]));
    step2[22] = ((step1[22]).wrapping_neg().wrapping_add(step1[23]));
    step2[23] = (step1[22].wrapping_add(step1[23]));
    step2[24] = (step1[24].wrapping_add(step1[25]));
    step2[25] = (step1[24].wrapping_sub(step1[25]));
    step2[26] = ((step1[26]).wrapping_neg().wrapping_add(step1[27]));
    step2[27] = (step1[26].wrapping_add(step1[27]));
    step2[28] = (step1[28].wrapping_add(step1[29]));
    step2[29] = (step1[28].wrapping_sub(step1[29]));
    step2[30] = ((step1[30]).wrapping_neg().wrapping_add(step1[31]));
    step2[31] = (step1[30].wrapping_add(step1[31]));
    step1[0] = (step2[0]);
    step1[1] = (step2[1]);
    step1[2] = (step2[2]);
    step1[3] = (step2[3]);
    temp1 = step2[4]
        .wrapping_mul(COSPI_28)
        .wrapping_sub(step2[7].wrapping_mul(COSPI_4));
    temp2 = step2[4]
        .wrapping_mul(COSPI_4)
        .wrapping_add(step2[7].wrapping_mul(COSPI_28));
    step1[4] = (rs(temp1));
    step1[7] = (rs(temp2));
    temp1 = step2[5]
        .wrapping_mul(COSPI_12)
        .wrapping_sub(step2[6].wrapping_mul(COSPI_20));
    temp2 = step2[5]
        .wrapping_mul(COSPI_20)
        .wrapping_add(step2[6].wrapping_mul(COSPI_12));
    step1[5] = (rs(temp1));
    step1[6] = (rs(temp2));
    step1[8] = (step2[8].wrapping_add(step2[9]));
    step1[9] = (step2[8].wrapping_sub(step2[9]));
    step1[10] = ((step2[10]).wrapping_neg().wrapping_add(step2[11]));
    step1[11] = (step2[10].wrapping_add(step2[11]));
    step1[12] = (step2[12].wrapping_add(step2[13]));
    step1[13] = (step2[12].wrapping_sub(step2[13]));
    step1[14] = ((step2[14]).wrapping_neg().wrapping_add(step2[15]));
    step1[15] = (step2[14].wrapping_add(step2[15]));
    step1[16] = (step2[16]);
    step1[31] = (step2[31]);
    temp1 = (step2[17])
        .wrapping_neg()
        .wrapping_mul(COSPI_4)
        .wrapping_add(step2[30].wrapping_mul(COSPI_28));
    temp2 = step2[17]
        .wrapping_mul(COSPI_28)
        .wrapping_add(step2[30].wrapping_mul(COSPI_4));
    step1[17] = (rs(temp1));
    step1[30] = (rs(temp2));
    temp1 = (step2[18])
        .wrapping_neg()
        .wrapping_mul(COSPI_28)
        .wrapping_sub(step2[29].wrapping_mul(COSPI_4));
    temp2 = (step2[18])
        .wrapping_neg()
        .wrapping_mul(COSPI_4)
        .wrapping_add(step2[29].wrapping_mul(COSPI_28));
    step1[18] = (rs(temp1));
    step1[29] = (rs(temp2));
    step1[19] = (step2[19]);
    step1[20] = (step2[20]);
    temp1 = (step2[21])
        .wrapping_neg()
        .wrapping_mul(COSPI_20)
        .wrapping_add(step2[26].wrapping_mul(COSPI_12));
    temp2 = step2[21]
        .wrapping_mul(COSPI_12)
        .wrapping_add(step2[26].wrapping_mul(COSPI_20));
    step1[21] = (rs(temp1));
    step1[26] = (rs(temp2));
    temp1 = (step2[22])
        .wrapping_neg()
        .wrapping_mul(COSPI_12)
        .wrapping_sub(step2[25].wrapping_mul(COSPI_20));
    temp2 = (step2[22])
        .wrapping_neg()
        .wrapping_mul(COSPI_20)
        .wrapping_add(step2[25].wrapping_mul(COSPI_12));
    step1[22] = (rs(temp1));
    step1[25] = (rs(temp2));
    step1[23] = (step2[23]);
    step1[24] = (step2[24]);
    step1[27] = (step2[27]);
    step1[28] = (step2[28]);
    temp1 = (step1[0].wrapping_add(step1[1])).wrapping_mul(COSPI_16);
    temp2 = (step1[0].wrapping_sub(step1[1])).wrapping_mul(COSPI_16);
    step2[0] = (rs(temp1));
    step2[1] = (rs(temp2));
    temp1 = step1[2]
        .wrapping_mul(COSPI_24)
        .wrapping_sub(step1[3].wrapping_mul(COSPI_8));
    temp2 = step1[2]
        .wrapping_mul(COSPI_8)
        .wrapping_add(step1[3].wrapping_mul(COSPI_24));
    step2[2] = (rs(temp1));
    step2[3] = (rs(temp2));
    step2[4] = (step1[4].wrapping_add(step1[5]));
    step2[5] = (step1[4].wrapping_sub(step1[5]));
    step2[6] = ((step1[6]).wrapping_neg().wrapping_add(step1[7]));
    step2[7] = (step1[6].wrapping_add(step1[7]));
    step2[8] = (step1[8]);
    step2[15] = (step1[15]);
    temp1 = (step1[9])
        .wrapping_neg()
        .wrapping_mul(COSPI_8)
        .wrapping_add(step1[14].wrapping_mul(COSPI_24));
    temp2 = step1[9]
        .wrapping_mul(COSPI_24)
        .wrapping_add(step1[14].wrapping_mul(COSPI_8));
    step2[9] = (rs(temp1));
    step2[14] = (rs(temp2));
    temp1 = (step1[10])
        .wrapping_neg()
        .wrapping_mul(COSPI_24)
        .wrapping_sub(step1[13].wrapping_mul(COSPI_8));
    temp2 = (step1[10])
        .wrapping_neg()
        .wrapping_mul(COSPI_8)
        .wrapping_add(step1[13].wrapping_mul(COSPI_24));
    step2[10] = (rs(temp1));
    step2[13] = (rs(temp2));
    step2[11] = (step1[11]);
    step2[12] = (step1[12]);
    step2[16] = (step1[16].wrapping_add(step1[19]));
    step2[17] = (step1[17].wrapping_add(step1[18]));
    step2[18] = (step1[17].wrapping_sub(step1[18]));
    step2[19] = (step1[16].wrapping_sub(step1[19]));
    step2[20] = ((step1[20]).wrapping_neg().wrapping_add(step1[23]));
    step2[21] = ((step1[21]).wrapping_neg().wrapping_add(step1[22]));
    step2[22] = (step1[21].wrapping_add(step1[22]));
    step2[23] = (step1[20].wrapping_add(step1[23]));
    step2[24] = (step1[24].wrapping_add(step1[27]));
    step2[25] = (step1[25].wrapping_add(step1[26]));
    step2[26] = (step1[25].wrapping_sub(step1[26]));
    step2[27] = (step1[24].wrapping_sub(step1[27]));
    step2[28] = ((step1[28]).wrapping_neg().wrapping_add(step1[31]));
    step2[29] = ((step1[29]).wrapping_neg().wrapping_add(step1[30]));
    step2[30] = (step1[29].wrapping_add(step1[30]));
    step2[31] = (step1[28].wrapping_add(step1[31]));
    step1[0] = (step2[0].wrapping_add(step2[3]));
    step1[1] = (step2[1].wrapping_add(step2[2]));
    step1[2] = (step2[1].wrapping_sub(step2[2]));
    step1[3] = (step2[0].wrapping_sub(step2[3]));
    step1[4] = (step2[4]);
    temp1 = (step2[6].wrapping_sub(step2[5])).wrapping_mul(COSPI_16);
    temp2 = (step2[5].wrapping_add(step2[6])).wrapping_mul(COSPI_16);
    step1[5] = (rs(temp1));
    step1[6] = (rs(temp2));
    step1[7] = (step2[7]);
    step1[8] = (step2[8].wrapping_add(step2[11]));
    step1[9] = (step2[9].wrapping_add(step2[10]));
    step1[10] = (step2[9].wrapping_sub(step2[10]));
    step1[11] = (step2[8].wrapping_sub(step2[11]));
    step1[12] = ((step2[12]).wrapping_neg().wrapping_add(step2[15]));
    step1[13] = ((step2[13]).wrapping_neg().wrapping_add(step2[14]));
    step1[14] = (step2[13].wrapping_add(step2[14]));
    step1[15] = (step2[12].wrapping_add(step2[15]));
    step1[16] = (step2[16]);
    step1[17] = (step2[17]);
    temp1 = (step2[18])
        .wrapping_neg()
        .wrapping_mul(COSPI_8)
        .wrapping_add(step2[29].wrapping_mul(COSPI_24));
    temp2 = step2[18]
        .wrapping_mul(COSPI_24)
        .wrapping_add(step2[29].wrapping_mul(COSPI_8));
    step1[18] = (rs(temp1));
    step1[29] = (rs(temp2));
    temp1 = (step2[19])
        .wrapping_neg()
        .wrapping_mul(COSPI_8)
        .wrapping_add(step2[28].wrapping_mul(COSPI_24));
    temp2 = step2[19]
        .wrapping_mul(COSPI_24)
        .wrapping_add(step2[28].wrapping_mul(COSPI_8));
    step1[19] = (rs(temp1));
    step1[28] = (rs(temp2));
    temp1 = (step2[20])
        .wrapping_neg()
        .wrapping_mul(COSPI_24)
        .wrapping_sub(step2[27].wrapping_mul(COSPI_8));
    temp2 = (step2[20])
        .wrapping_neg()
        .wrapping_mul(COSPI_8)
        .wrapping_add(step2[27].wrapping_mul(COSPI_24));
    step1[20] = (rs(temp1));
    step1[27] = (rs(temp2));
    temp1 = (step2[21])
        .wrapping_neg()
        .wrapping_mul(COSPI_24)
        .wrapping_sub(step2[26].wrapping_mul(COSPI_8));
    temp2 = (step2[21])
        .wrapping_neg()
        .wrapping_mul(COSPI_8)
        .wrapping_add(step2[26].wrapping_mul(COSPI_24));
    step1[21] = (rs(temp1));
    step1[26] = (rs(temp2));
    step1[22] = (step2[22]);
    step1[23] = (step2[23]);
    step1[24] = (step2[24]);
    step1[25] = (step2[25]);
    step1[30] = (step2[30]);
    step1[31] = (step2[31]);
    step2[0] = (step1[0].wrapping_add(step1[7]));
    step2[1] = (step1[1].wrapping_add(step1[6]));
    step2[2] = (step1[2].wrapping_add(step1[5]));
    step2[3] = (step1[3].wrapping_add(step1[4]));
    step2[4] = (step1[3].wrapping_sub(step1[4]));
    step2[5] = (step1[2].wrapping_sub(step1[5]));
    step2[6] = (step1[1].wrapping_sub(step1[6]));
    step2[7] = (step1[0].wrapping_sub(step1[7]));
    step2[8] = (step1[8]);
    step2[9] = (step1[9]);
    temp1 = ((step1[10]).wrapping_neg().wrapping_add(step1[13])).wrapping_mul(COSPI_16);
    temp2 = (step1[10].wrapping_add(step1[13])).wrapping_mul(COSPI_16);
    step2[10] = (rs(temp1));
    step2[13] = (rs(temp2));
    temp1 = ((step1[11]).wrapping_neg().wrapping_add(step1[12])).wrapping_mul(COSPI_16);
    temp2 = (step1[11].wrapping_add(step1[12])).wrapping_mul(COSPI_16);
    step2[11] = (rs(temp1));
    step2[12] = (rs(temp2));
    step2[14] = (step1[14]);
    step2[15] = (step1[15]);
    step2[16] = (step1[16].wrapping_add(step1[23]));
    step2[17] = (step1[17].wrapping_add(step1[22]));
    step2[18] = (step1[18].wrapping_add(step1[21]));
    step2[19] = (step1[19].wrapping_add(step1[20]));
    step2[20] = (step1[19].wrapping_sub(step1[20]));
    step2[21] = (step1[18].wrapping_sub(step1[21]));
    step2[22] = (step1[17].wrapping_sub(step1[22]));
    step2[23] = (step1[16].wrapping_sub(step1[23]));
    step2[24] = ((step1[24]).wrapping_neg().wrapping_add(step1[31]));
    step2[25] = ((step1[25]).wrapping_neg().wrapping_add(step1[30]));
    step2[26] = ((step1[26]).wrapping_neg().wrapping_add(step1[29]));
    step2[27] = ((step1[27]).wrapping_neg().wrapping_add(step1[28]));
    step2[28] = (step1[27].wrapping_add(step1[28]));
    step2[29] = (step1[26].wrapping_add(step1[29]));
    step2[30] = (step1[25].wrapping_add(step1[30]));
    step2[31] = (step1[24].wrapping_add(step1[31]));
    step1[0] = (step2[0].wrapping_add(step2[15]));
    step1[1] = (step2[1].wrapping_add(step2[14]));
    step1[2] = (step2[2].wrapping_add(step2[13]));
    step1[3] = (step2[3].wrapping_add(step2[12]));
    step1[4] = (step2[4].wrapping_add(step2[11]));
    step1[5] = (step2[5].wrapping_add(step2[10]));
    step1[6] = (step2[6].wrapping_add(step2[9]));
    step1[7] = (step2[7].wrapping_add(step2[8]));
    step1[8] = (step2[7].wrapping_sub(step2[8]));
    step1[9] = (step2[6].wrapping_sub(step2[9]));
    step1[10] = (step2[5].wrapping_sub(step2[10]));
    step1[11] = (step2[4].wrapping_sub(step2[11]));
    step1[12] = (step2[3].wrapping_sub(step2[12]));
    step1[13] = (step2[2].wrapping_sub(step2[13]));
    step1[14] = (step2[1].wrapping_sub(step2[14]));
    step1[15] = (step2[0].wrapping_sub(step2[15]));
    step1[16] = (step2[16]);
    step1[17] = (step2[17]);
    step1[18] = (step2[18]);
    step1[19] = (step2[19]);
    temp1 = ((step2[20]).wrapping_neg().wrapping_add(step2[27])).wrapping_mul(COSPI_16);
    temp2 = (step2[20].wrapping_add(step2[27])).wrapping_mul(COSPI_16);
    step1[20] = (rs(temp1));
    step1[27] = (rs(temp2));
    temp1 = ((step2[21]).wrapping_neg().wrapping_add(step2[26])).wrapping_mul(COSPI_16);
    temp2 = (step2[21].wrapping_add(step2[26])).wrapping_mul(COSPI_16);
    step1[21] = (rs(temp1));
    step1[26] = (rs(temp2));
    temp1 = ((step2[22]).wrapping_neg().wrapping_add(step2[25])).wrapping_mul(COSPI_16);
    temp2 = (step2[22].wrapping_add(step2[25])).wrapping_mul(COSPI_16);
    step1[22] = (rs(temp1));
    step1[25] = (rs(temp2));
    temp1 = ((step2[23]).wrapping_neg().wrapping_add(step2[24])).wrapping_mul(COSPI_16);
    temp2 = (step2[23].wrapping_add(step2[24])).wrapping_mul(COSPI_16);
    step1[23] = (rs(temp1));
    step1[24] = (rs(temp2));
    step1[28] = (step2[28]);
    step1[29] = (step2[29]);
    step1[30] = (step2[30]);
    step1[31] = (step2[31]);
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

