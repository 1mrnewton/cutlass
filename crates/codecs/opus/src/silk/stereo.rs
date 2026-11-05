//! Stereo prediction: the mid/side predictor weights and the conversion
//! of mid and side to left and right (RFC 6716 sections 4.2.7.1 and
//! 4.2.8; reference `stereo_decode_pred.c`, `stereo_MS_to_LR.c`).

use super::fixed::{lshift, rshift_round, sat16, smlabb, smlawb, smulbb, smulwb};
use super::tables::{
    STEREO_ONLY_CODE_MID_ICDF, STEREO_PRED_JOINT_ICDF, STEREO_PRED_QUANT_Q13, UNIFORM3_ICDF,
    UNIFORM5_ICDF,
};
use crate::range::RangeDecoder;

/// `STEREO_INTERP_LEN_MS`: predictor interpolation at the frame start.
const STEREO_INTERP_LEN_MS: usize = 8;

/// `stereo_dec_state`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StereoState {
    /// Predictors of the previous frame, Q13.
    pub pred_prev_q13: [i16; 2],
    /// The last two mid samples, output one frame later.
    pub s_mid: [i16; 2],
    /// The last two side samples.
    pub s_side: [i16; 2],
}

/// `silk_stereo_decode_pred`: the two mid-to-side predictors in Q13.
pub(crate) fn decode_pred(rd: &mut RangeDecoder<'_>) -> [i32; 2] {
    let joint = rd.dec_icdf(&STEREO_PRED_JOINT_ICDF, 8);
    let mut ix = [[0usize; 3]; 2];
    ix[0][2] = joint / 5;
    ix[1][2] = joint - 5 * ix[0][2];
    for row in &mut ix {
        row[0] = rd.dec_icdf(&UNIFORM3_ICDF, 8);
        row[1] = rd.dec_icdf(&UNIFORM5_ICDF, 8);
    }
    let mut pred_q13 = [0i32; 2];
    for (n, row) in ix.iter().enumerate() {
        let i = (row[0] + 3 * row[2]).min(STEREO_PRED_QUANT_Q13.len() - 2);
        let low_q13 = i32::from(STEREO_PRED_QUANT_Q13[i]);
        let step_q13 = smulwb(i32::from(STEREO_PRED_QUANT_Q13[i + 1]) - low_q13, 6554);
        pred_q13[n] = smlabb(low_q13, step_q13, 2 * row[1] as i32 + 1);
    }
    pred_q13[0] -= pred_q13[1];
    pred_q13
}

/// `silk_stereo_decode_mid_only`: whether this frame codes only the mid
/// channel.
pub(crate) fn decode_mid_only(rd: &mut RangeDecoder<'_>) -> bool {
    rd.dec_icdf(&STEREO_ONLY_CODE_MID_ICDF, 8) != 0
}

/// `silk_stereo_MS_to_LR`. `x1` and `x2` hold two samples of history
/// followed by `frame_length` mid and side samples; on return
/// `x1[1..=frame_length]` and `x2[1..=frame_length]` are left and right.
pub(crate) fn ms_to_lr(
    state: &mut StereoState,
    x1: &mut [i16],
    x2: &mut [i16],
    pred_q13: [i32; 2],
    fs_khz: usize,
    frame_length: usize,
) {
    let n_total = frame_length
        .min(x1.len().saturating_sub(2))
        .min(x2.len().saturating_sub(2));
    if n_total == 0 {
        return;
    }
    x1[..2].copy_from_slice(&state.s_mid);
    x2[..2].copy_from_slice(&state.s_side);
    state.s_mid.copy_from_slice(&x1[n_total..n_total + 2]);
    state.s_side.copy_from_slice(&x2[n_total..n_total + 2]);

    // Interpolate the predictors over the first 8 ms, then hold them.
    let interp_len = (STEREO_INTERP_LEN_MS * fs_khz).clamp(1, n_total);
    let mut pred0_q13 = i32::from(state.pred_prev_q13[0]);
    let mut pred1_q13 = i32::from(state.pred_prev_q13[1]);
    let denom_q16 = (1 << 16) / (STEREO_INTERP_LEN_MS * fs_khz).max(1) as i32;
    let delta0_q13 = rshift_round(smulbb(pred_q13[0] - pred0_q13, denom_q16), 16);
    let delta1_q13 = rshift_round(smulbb(pred_q13[1] - pred1_q13, denom_q16), 16);
    for n in 0..n_total {
        if n < interp_len {
            pred0_q13 += delta0_q13;
            pred1_q13 += delta1_q13;
        } else if n == interp_len {
            pred0_q13 = pred_q13[0];
            pred1_q13 = pred_q13[1];
        }
        let mid2 = i32::from(x1[n]) + i32::from(x1[n + 2]) + (i32::from(x1[n + 1]) << 1);
        let mut sum = lshift(mid2, 9); // Q11
        sum = smlawb(i32::from(x2[n + 1]) << 8, sum, pred0_q13); // Q8
        sum = smlawb(sum, i32::from(x1[n + 1]) << 11, pred1_q13); // Q8
        x2[n + 1] = sat16(rshift_round(sum, 8));
    }
    state.pred_prev_q13 = [pred_q13[0] as i16, pred_q13[1] as i16];

    for n in 1..=n_total {
        let m = i32::from(x1[n]);
        let s = i32::from(x2[n]);
        x1[n] = sat16(m + s);
        x2[n] = sat16(m - s);
    }
}

#[cfg(test)]
mod tests {
    use super::{StereoState, ms_to_lr};

    #[test]
    fn zero_predictors_turn_mid_and_side_into_left_and_right() {
        let mut state = StereoState::default();
        let n = 160;
        let mut x1 = vec![0i16; n + 2];
        let mut x2 = vec![0i16; n + 2];
        for i in 0..n {
            x1[i + 2] = 1000;
            x2[i + 2] = 200;
        }
        ms_to_lr(&mut state, &mut x1, &mut x2, [0, 0], 16, n);
        // Output is one sample late: x[1] is the first history sample.
        assert_eq!((x1[1], x2[1]), (0, 0));
        assert_eq!((x1[2], x2[2]), (1200, 800));
        assert_eq!((x1[n], x2[n]), (1200, 800));
        assert_eq!(state.s_mid, [1000, 1000]);
    }

    #[test]
    fn a_full_side_predictor_moves_mid_into_one_side() {
        let mut state = StereoState {
            pred_prev_q13: [0, 8192],
            ..StereoState::default()
        };
        let n = 80;
        let mut x1 = vec![4000i16; n + 2];
        let mut x2 = vec![0i16; n + 2];
        ms_to_lr(&mut state, &mut x1, &mut x2, [0, 8192], 8, n);
        // side = mid * 1.0, so left = 2 mid and right = 0.
        assert!((i32::from(x1[40]) - 8000).abs() <= 2, "{}", x1[40]);
        assert!(x2[40].abs() <= 2);
    }
}
