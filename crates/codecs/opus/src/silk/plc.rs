//! Packet-loss concealment: extrapolates a lost frame from the last good
//! one and fades the next good frame in (RFC 6716 section 4.2.8 leaves PLC
//! to the implementation; this is the reference `PLC.c`).

use super::channel::{
    ChannelState, DecoderControl, LTP_ORDER, MAX_SUB_FRAME_LENGTH, lpc_analysis_filter,
};
use super::fixed::{
    clz32, div32, inverse32_varq, lshift, rand, rshift_round, sat16, smlawb, smulbb, smulwb,
    smulww, sqrt_approx, sum_sqr_shift,
};
use super::indices::{MAX_NB_SUBFR, TYPE_VOICED};
use super::nlsf::{MAX_LPC_ORDER, bwexpander, lpc_inverse_pred_gain};
use super::pulses::MAX_FRAME_LENGTH;

/// `BWE_COEF` in Q16 (0.99).
const BWE_COEF_Q16: i32 = 64881;
/// `V_PITCH_GAIN_START_MIN_Q14` (0.7).
const V_PITCH_GAIN_START_MIN_Q14: i32 = 11469;
/// `V_PITCH_GAIN_START_MAX_Q14` (0.95).
const V_PITCH_GAIN_START_MAX_Q14: i32 = 15565;
/// `MAX_PITCH_LAG_MS`.
const MAX_PITCH_LAG_MS: i32 = 18;
/// `RAND_BUF_SIZE`.
const RAND_BUF_SIZE: usize = 128;
/// `LOG2_INV_LPC_GAIN_HIGH_THRES`.
const LOG2_INV_LPC_GAIN_HIGH_THRES: u32 = 3;
/// `LOG2_INV_LPC_GAIN_LOW_THRES`.
const LOG2_INV_LPC_GAIN_LOW_THRES: u32 = 8;
/// `PITCH_DRIFT_FAC_Q16` (0.01).
const PITCH_DRIFT_FAC_Q16: i32 = 655;
/// Attenuation of the harmonic part per lost frame (0.99, 0.95).
const HARM_ATT_Q15: [i32; 2] = [32440, 31130];
/// Attenuation of the noise part after voiced frames (0.95, 0.8).
const PLC_RAND_ATTENUATE_V_Q15: [i32; 2] = [31130, 26214];
/// Attenuation of the noise part after unvoiced frames (0.99, 0.9).
const PLC_RAND_ATTENUATE_UV_Q15: [i32; 2] = [32440, 29491];

/// `silk_PLC_struct`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PlcState {
    pub pitch_l_q8: i32,
    pub ltp_coef_q14: [i16; LTP_ORDER],
    pub prev_lpc_q12: [i16; MAX_LPC_ORDER],
    pub last_frame_lost: bool,
    pub rand_seed: i32,
    pub rand_scale_q14: i16,
    pub conc_energy: i32,
    pub conc_energy_shift: i32,
    pub prev_ltp_scale_q14: i32,
    pub prev_gain_q16: [i32; 2],
    pub fs_khz: i32,
    pub nb_subfr: usize,
    pub subfr_length: usize,
}

/// `silk_PLC_Reset`.
pub(crate) fn reset(st: &mut ChannelState) {
    st.plc.pitch_l_q8 = lshift(st.frame_length as i32, 8 - 1);
    st.plc.prev_gain_q16 = [1 << 16; 2];
    st.plc.subfr_length = 20;
    st.plc.nb_subfr = 2;
}

/// The rate check at the top of `silk_PLC`.
fn follow_rate(st: &mut ChannelState) {
    if st.fs_khz != st.plc.fs_khz {
        reset(st);
        st.plc.fs_khz = st.fs_khz;
    }
}

/// `silk_PLC` for a good frame: remembers what concealment will need.
pub(crate) fn update(st: &mut ChannelState, ctrl: &DecoderControl) {
    follow_rate(st);
    let nb_subfr = st.nb_subfr.clamp(2, MAX_NB_SUBFR);
    let plc = &mut st.plc;
    st.prev_signal_type = st.indices.signal_type;
    let mut ltp_gain_q14: i32 = 0;
    if st.indices.signal_type == TYPE_VOICED {
        // Use the last subframe that holds a full pitch pulse.
        let last_lag = ctrl.pitch_l[nb_subfr - 1];
        let mut j = 0;
        while (j * st.subfr_length) < usize::try_from(last_lag).unwrap_or(0) {
            if j == nb_subfr {
                break;
            }
            let sf = nb_subfr - 1 - j;
            let taps = &ctrl.ltp_coef_q14[sf * LTP_ORDER..(sf + 1) * LTP_ORDER];
            let temp: i32 = taps.iter().map(|&t| i32::from(t)).sum();
            if temp > ltp_gain_q14 {
                ltp_gain_q14 = temp;
                plc.ltp_coef_q14.copy_from_slice(taps);
                plc.pitch_l_q8 = lshift(ctrl.pitch_l[sf], 8);
            }
            j += 1;
        }
        plc.ltp_coef_q14 = [0; LTP_ORDER];
        plc.ltp_coef_q14[LTP_ORDER / 2] = ltp_gain_q14 as i16;

        // Keep the LTP gain between 0.7 and 0.95.
        if ltp_gain_q14 < V_PITCH_GAIN_START_MIN_Q14 {
            let scale_q10 = div32(V_PITCH_GAIN_START_MIN_Q14 << 10, ltp_gain_q14.max(1));
            for c in plc.ltp_coef_q14.iter_mut() {
                *c = (smulbb(i32::from(*c), scale_q10) >> 10) as i16;
            }
        } else if ltp_gain_q14 > V_PITCH_GAIN_START_MAX_Q14 {
            let scale_q14 = div32(V_PITCH_GAIN_START_MAX_Q14 << 14, ltp_gain_q14.max(1));
            for c in plc.ltp_coef_q14.iter_mut() {
                *c = (smulbb(i32::from(*c), scale_q14) >> 14) as i16;
            }
        }
    } else {
        plc.pitch_l_q8 = lshift(smulbb(st.fs_khz, 18), 8);
        plc.ltp_coef_q14 = [0; LTP_ORDER];
    }
    let order = st.lpc_order;
    plc.prev_lpc_q12[..order].copy_from_slice(&ctrl.pred_coef_q12[1][..order]);
    plc.prev_ltp_scale_q14 = ctrl.ltp_scale_q14;
    plc.prev_gain_q16 = [ctrl.gains_q16[nb_subfr - 2], ctrl.gains_q16[nb_subfr - 1]];
    plc.subfr_length = st.subfr_length;
    plc.nb_subfr = nb_subfr;
}

/// `silk_PLC` for a lost frame (`silk_PLC_conceal`): an LTP extrapolation
/// of the last pitch period plus attenuated noise from the last
/// excitation, through the last LPC filter.
pub(crate) fn conceal(st: &mut ChannelState, ctrl: &mut DecoderControl, frame: &mut [i16]) {
    follow_rate(st);
    let order = st.lpc_order;
    let ltp_mem_length = st.ltp_mem_length;
    let frame_length = frame.len().min(MAX_FRAME_LENGTH);
    let subfr_length = st.subfr_length.min(MAX_SUB_FRAME_LENGTH);
    let fs_khz = st.fs_khz;
    let loss_index = usize::from(st.loss_cnt > 0);
    let prev_voiced = st.prev_signal_type == TYPE_VOICED;
    let plc = &mut st.plc;

    let prev_gain_q10 = [plc.prev_gain_q16[0] >> 6, plc.prev_gain_q16[1] >> 6];
    if st.first_frame_after_reset {
        plc.prev_lpc_q12 = [0; MAX_LPC_ORDER];
    }

    // Of the last two subframes' scaled excitation, the quieter one seeds
    // the noise.
    let plc_subfr = plc.subfr_length.min(MAX_SUB_FRAME_LENGTH);
    let plc_nb_subfr = plc.nb_subfr.clamp(2, MAX_NB_SUBFR);
    let mut exc_buf = [0i16; 2 * MAX_SUB_FRAME_LENGTH];
    for k in 0..2 {
        for i in 0..plc_subfr {
            let e = st.exc_q14[i + (k + plc_nb_subfr - 2) * plc_subfr];
            exc_buf[k * plc_subfr + i] = sat16(smulww(e, prev_gain_q10[k]) >> 8);
        }
    }
    let (energy1, shift1) = sum_sqr_shift(&exc_buf[..plc_subfr]);
    let (energy2, shift2) = sum_sqr_shift(&exc_buf[plc_subfr..2 * plc_subfr]);
    let rand_start = if (energy1 >> shift2) < (energy2 >> shift1) {
        ((plc_nb_subfr - 1) * plc_subfr).saturating_sub(RAND_BUF_SIZE)
    } else {
        (plc_nb_subfr * plc_subfr).saturating_sub(RAND_BUF_SIZE)
    };

    let harm_gain_q15 = HARM_ATT_Q15[loss_index];
    let mut rand_gain_q15 = if prev_voiced {
        PLC_RAND_ATTENUATE_V_Q15[loss_index]
    } else {
        PLC_RAND_ATTENUATE_UV_Q15[loss_index]
    };

    // Chirp the last filter a little more with every lost frame.
    bwexpander(&mut plc.prev_lpc_q12[..order], BWE_COEF_Q16);
    let a_q12 = plc.prev_lpc_q12;

    let mut rand_scale_q14 = plc.rand_scale_q14;
    if st.loss_cnt == 0 {
        // First lost frame.
        rand_scale_q14 = 1 << 14;
        if prev_voiced {
            // Less noise when the pitch carries the signal.
            for &b in &plc.ltp_coef_q14 {
                rand_scale_q14 = rand_scale_q14.wrapping_sub(b);
            }
            rand_scale_q14 = rand_scale_q14.max(3277);
            rand_scale_q14 =
                (smulbb(i32::from(rand_scale_q14), plc.prev_ltp_scale_q14) >> 14) as i16;
        } else {
            // Less noise for unvoiced frames with a high LPC gain.
            let inv_gain_q30 = lpc_inverse_pred_gain(&a_q12[..order]);
            let down_scale_q30 = inv_gain_q30.clamp(
                (1 << 30) >> LOG2_INV_LPC_GAIN_LOW_THRES,
                (1 << 30) >> LOG2_INV_LPC_GAIN_HIGH_THRES,
            ) << LOG2_INV_LPC_GAIN_HIGH_THRES;
            rand_gain_q15 = smulwb(down_scale_q30, rand_gain_q15) >> 14;
        }
    }

    let mut rand_seed = plc.rand_seed;
    let max_lag = ltp_mem_length.saturating_sub(order + LTP_ORDER / 2);
    let lag_of = |pitch_l_q8: i32| {
        usize::try_from(rshift_round(pitch_l_q8, 8))
            .unwrap_or(0)
            .min(max_lag)
    };
    let mut lag = lag_of(plc.pitch_l_q8);
    let mut ltp_buf_idx = ltp_mem_length;

    // Rewhiten the LTP state with the chirped filter.
    let idx = ltp_mem_length - lag - order - LTP_ORDER / 2;
    let mut s_ltp = [0i16; MAX_FRAME_LENGTH];
    lpc_analysis_filter(
        &mut s_ltp[idx..ltp_mem_length],
        &st.out_buf[idx..ltp_mem_length],
        &a_q12[..order],
    );
    let inv_gain_q30 = inverse32_varq(plc.prev_gain_q16[1], 46).min(i32::MAX >> 1);
    let mut s_ltp_q14 = [0i32; 2 * MAX_FRAME_LENGTH + 8];
    for i in idx + order..ltp_mem_length {
        s_ltp_q14[i] = smulwb(inv_gain_q30, i32::from(s_ltp[i]));
    }

    // LTP synthesis with noise excitation.
    let b_q14 = &mut plc.ltp_coef_q14;
    for _ in 0..st.nb_subfr.min(MAX_NB_SUBFR) {
        for _ in 0..subfr_length {
            let pred = ltp_buf_idx + LTP_ORDER / 2 - lag;
            let mut ltp_pred_q12: i32 = 2;
            for (j, &b) in b_q14.iter().enumerate() {
                ltp_pred_q12 = smlawb(ltp_pred_q12, s_ltp_q14[pred - j], i32::from(b));
            }
            rand_seed = rand(rand_seed);
            let r = ((rand_seed >> 25) as usize) & (RAND_BUF_SIZE - 1);
            let noise = st.exc_q14[(rand_start + r).min(MAX_FRAME_LENGTH - 1)];
            s_ltp_q14[ltp_buf_idx] =
                lshift(smlawb(ltp_pred_q12, noise, i32::from(rand_scale_q14)), 2);
            ltp_buf_idx += 1;
        }
        // Fade the harmonic and noise parts, and let the pitch drift up.
        for b in b_q14.iter_mut() {
            *b = (smulbb(harm_gain_q15, i32::from(*b)) >> 15) as i16;
        }
        rand_scale_q14 = (smulbb(i32::from(rand_scale_q14), rand_gain_q15) >> 15) as i16;
        plc.pitch_l_q8 = smlawb(plc.pitch_l_q8, plc.pitch_l_q8, PITCH_DRIFT_FAC_Q16);
        plc.pitch_l_q8 = plc
            .pitch_l_q8
            .min(lshift(smulbb(MAX_PITCH_LAG_MS, fs_khz), 8));
        lag = lag_of(plc.pitch_l_q8);
    }

    // LPC synthesis.
    let base = ltp_mem_length - MAX_LPC_ORDER;
    s_ltp_q14[base..ltp_mem_length].copy_from_slice(&st.s_lpc_q14_buf);
    let s_lpc = &mut s_ltp_q14[base..];
    for (i, out) in frame.iter_mut().enumerate().take(frame_length) {
        let mut lpc_pred_q10 = (order >> 1) as i32;
        for (j, &a) in a_q12.iter().take(order).enumerate() {
            lpc_pred_q10 = smlawb(lpc_pred_q10, s_lpc[MAX_LPC_ORDER + i - j - 1], i32::from(a));
        }
        let v = s_lpc[MAX_LPC_ORDER + i].wrapping_add(lshift(lpc_pred_q10, 4));
        s_lpc[MAX_LPC_ORDER + i] = v;
        *out = sat16(rshift_round(smulww(v, prev_gain_q10[1]), 8));
    }
    st.s_lpc_q14_buf
        .copy_from_slice(&s_lpc[frame_length..frame_length + MAX_LPC_ORDER]);

    plc.rand_seed = rand_seed;
    plc.rand_scale_q14 = rand_scale_q14;
    ctrl.pitch_l = [lag as i32; MAX_NB_SUBFR];
    st.loss_cnt += 1;
}

/// `silk_PLC_glue_frames`: measures concealed frames, and fades in a good
/// frame after a loss when it is louder than the concealment.
pub(crate) fn glue_frames(st: &mut ChannelState, frame: &mut [i16]) {
    let plc = &mut st.plc;
    if st.loss_cnt != 0 {
        (plc.conc_energy, plc.conc_energy_shift) = sum_sqr_shift(frame);
        plc.last_frame_lost = true;
        return;
    }
    if plc.last_frame_lost {
        let (mut energy, energy_shift) = sum_sqr_shift(frame);
        if energy_shift > plc.conc_energy_shift {
            plc.conc_energy >>= (energy_shift - plc.conc_energy_shift).min(31);
        } else if energy_shift < plc.conc_energy_shift {
            energy >>= (plc.conc_energy_shift - energy_shift).min(31);
        }
        if energy > plc.conc_energy {
            let lz = clz32(plc.conc_energy) - 1;
            plc.conc_energy = lshift(plc.conc_energy, lz.clamp(0, 31) as u32);
            energy >>= (24 - lz).clamp(0, 31);
            let frac_q24 = div32(plc.conc_energy, energy.max(1));
            let mut gain_q16 = lshift(sqrt_approx(frac_q24), 4);
            let length = i32::try_from(frame.len()).unwrap_or(i32::MAX).max(1);
            // Four times steeper than linear, so onsets after DTX survive.
            let slope_q16 = lshift(div32((1 << 16) - gain_q16, length), 2);
            for s in frame.iter_mut() {
                *s = smulwb(gain_q16, i32::from(*s)) as i16;
                gain_q16 = gain_q16.wrapping_add(slope_q16);
                if gain_q16 > 1 << 16 {
                    break;
                }
            }
        }
    }
    plc.last_frame_lost = false;
}
