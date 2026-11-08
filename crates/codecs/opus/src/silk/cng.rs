//! Comfort noise generation: tracks the background spectrum and level
//! during inactive frames and adds matching noise to concealed frames
//! (reference `CNG.c`).

use super::channel::{ChannelState, DecoderControl};
use super::fixed::{rand, rshift_round, sat16, smlawb, smulwb, smulww};
use super::indices::TYPE_NO_VOICE_ACTIVITY;
use super::nlsf::{MAX_LPC_ORDER, nlsf2a};
use super::pulses::MAX_FRAME_LENGTH;

/// `CNG_BUF_MASK_MAX`.
const CNG_BUF_MASK_MAX: usize = 255;
/// `CNG_GAIN_SMTH_Q16`.
const CNG_GAIN_SMTH_Q16: i32 = 4634;
/// `CNG_NLSF_SMTH_Q16`.
const CNG_NLSF_SMTH_Q16: i32 = 16348;

/// `silk_CNG_struct`.
#[derive(Clone, Debug)]
pub(crate) struct CngState {
    pub exc_buf_q14: [i32; MAX_FRAME_LENGTH],
    pub smth_nlsf_q15: [i16; MAX_LPC_ORDER],
    pub synth_state: [i32; MAX_LPC_ORDER],
    pub smth_gain_q16: i32,
    pub rand_seed: i32,
    pub fs_khz: i32,
}

impl CngState {
    pub(crate) fn new() -> Self {
        Self {
            exc_buf_q14: [0; MAX_FRAME_LENGTH],
            smth_nlsf_q15: [0; MAX_LPC_ORDER],
            synth_state: [0; MAX_LPC_ORDER],
            smth_gain_q16: 0,
            rand_seed: 0,
            fs_khz: 0,
        }
    }
}

/// `silk_CNG_Reset`: evenly spaced NLSFs, zero gain.
pub(crate) fn reset(st: &mut ChannelState) {
    let order = st.lpc_order.min(MAX_LPC_ORDER);
    let step = i32::from(i16::MAX) / (order as i32 + 1);
    let mut acc = 0;
    for v in st.cng.smth_nlsf_q15.iter_mut().take(order) {
        acc += step;
        *v = acc as i16;
    }
    st.cng.smth_gain_q16 = 0;
    st.cng.rand_seed = 3_176_576;
}

/// `silk_CNG`: updates the noise model from an inactive good frame, and
/// adds comfort noise to a concealed frame.
pub(crate) fn apply(st: &mut ChannelState, ctrl: &DecoderControl, frame: &mut [i16]) {
    if st.fs_khz != st.cng.fs_khz {
        reset(st);
        st.cng.fs_khz = st.fs_khz;
    }
    let order = st.lpc_order.min(MAX_LPC_ORDER);
    let nb_subfr = st.nb_subfr.min(ctrl.gains_q16.len());
    let subfr_length = st.subfr_length;
    let cng = &mut st.cng;

    if st.loss_cnt == 0 && st.prev_signal_type == TYPE_NO_VOICE_ACTIVITY {
        // Smooth the NLSFs.
        for (s, &p) in cng
            .smth_nlsf_q15
            .iter_mut()
            .zip(&st.prev_nlsf_q15)
            .take(order)
        {
            let delta = i32::from(p) - i32::from(*s);
            *s = (i32::from(*s) + smulwb(delta, CNG_NLSF_SMTH_Q16)) as i16;
        }
        // Keep the excitation of the loudest subframe.
        let mut max_gain_q16 = 0;
        let mut subfr = 0;
        for (i, &g) in ctrl.gains_q16.iter().enumerate().take(nb_subfr) {
            if g > max_gain_q16 {
                max_gain_q16 = g;
                subfr = i;
            }
        }
        let keep = nb_subfr.saturating_sub(1) * subfr_length;
        if subfr_length + keep <= MAX_FRAME_LENGTH && (subfr + 1) * subfr_length <= MAX_FRAME_LENGTH
        {
            cng.exc_buf_q14.copy_within(0..keep, subfr_length);
            cng.exc_buf_q14[..subfr_length]
                .copy_from_slice(&st.exc_q14[subfr * subfr_length..(subfr + 1) * subfr_length]);
        }
        // Smooth the gains.
        for &g in ctrl.gains_q16.iter().take(nb_subfr) {
            cng.smth_gain_q16 = cng
                .smth_gain_q16
                .wrapping_add(smulwb(g.wrapping_sub(cng.smth_gain_q16), CNG_GAIN_SMTH_Q16));
        }
    }

    if st.loss_cnt != 0 {
        let length = frame.len().min(MAX_FRAME_LENGTH);
        // Noise excitation drawn at random from the stored excitation.
        let mut exc_mask = CNG_BUF_MASK_MAX;
        while exc_mask > length {
            exc_mask >>= 1;
        }
        let mut sig_q10 = [0i32; MAX_FRAME_LENGTH + MAX_LPC_ORDER];
        let mut seed = cng.rand_seed;
        for v in sig_q10[MAX_LPC_ORDER..MAX_LPC_ORDER + length].iter_mut() {
            seed = rand(seed);
            let idx = ((seed >> 24) as usize) & exc_mask;
            *v = i32::from(sat16(smulww(cng.exc_buf_q14[idx], cng.smth_gain_q16 >> 4)));
        }
        cng.rand_seed = seed;

        let mut a_q12 = [0i16; MAX_LPC_ORDER];
        nlsf2a(&mut a_q12[..order], &cng.smth_nlsf_q15[..order]);

        // Synthesis filter, added to the concealed signal.
        sig_q10[..MAX_LPC_ORDER].copy_from_slice(&cng.synth_state);
        for (i, out) in frame.iter_mut().enumerate().take(length) {
            let mut sum_q6 = (order >> 1) as i32;
            for (j, &a) in a_q12.iter().take(order).enumerate() {
                sum_q6 = smlawb(sum_q6, sig_q10[MAX_LPC_ORDER + i - j - 1], i32::from(a));
            }
            let at = MAX_LPC_ORDER + i;
            sig_q10[at] = sig_q10[at].wrapping_add(sum_q6.wrapping_shl(4));
            *out = sat16(i32::from(*out).saturating_add(rshift_round(sum_q6, 6)));
        }
        cng.synth_state
            .copy_from_slice(&sig_q10[length..length + MAX_LPC_ORDER]);
    } else {
        cng.synth_state[..order].fill(0);
    }
}
