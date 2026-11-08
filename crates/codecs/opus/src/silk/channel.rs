//! One SILK channel: its decoder state, the frame decoder, parameter
//! dequantization, and the LTP and LPC synthesis filters (RFC 6716
//! sections 4.2.7.4-4.2.7.9; reference `init_decoder.c`,
//! `decoder_set_fs.c`, `decode_frame.c`, `decode_parameters.c`,
//! `decode_pitch.c`, `gain_quant.c`, `decode_core.c`,
//! `LPC_analysis_filter.c`).

use super::cng::{self, CngState};
use super::fixed::{
    div32_varq, inverse32_varq, log2lin, lshift, rand, rshift_round, sat16, smlabb, smlawb, smulwb,
    smulww,
};
use super::indices::{
    CondCoding, EntropyHistory, IndexLayout, MAX_NB_SUBFR, SideInfo, TYPE_NO_VOICE_ACTIVITY,
    TYPE_VOICED, decode_indices,
};
use super::nlsf::{self, MAX_LPC_ORDER};
use super::plc::{self, PlcState};
use super::pulses::{MAX_FRAME_LENGTH, decode_pulses};
use super::resampler::Resampler;
use super::tables::{
    CB_LAGS_STAGE2, CB_LAGS_STAGE2_10_MS, CB_LAGS_STAGE3, CB_LAGS_STAGE3_10_MS, LTP_GAIN_VQ_0,
    LTP_GAIN_VQ_1, LTP_GAIN_VQ_2, LTP_SCALES_Q14, NLSF_CB_NB_MB, NLSF_CB_WB, NlsfCodebook,
    PITCH_CONTOUR_10_MS_ICDF, PITCH_CONTOUR_10_MS_NB_ICDF, PITCH_CONTOUR_ICDF,
    PITCH_CONTOUR_NB_ICDF, QUANTIZATION_OFFSETS_Q10, UNIFORM4_ICDF, UNIFORM6_ICDF, UNIFORM8_ICDF,
};
use crate::range::RangeDecoder;

/// `LTP_ORDER`: taps of the long-term predictor.
pub(crate) const LTP_ORDER: usize = 5;
/// Longest subframe: 5 ms at 16 kHz.
pub(crate) const MAX_SUB_FRAME_LENGTH: usize = 80;
/// `outBuf` length: the LTP memory plus two subframes.
const OUT_BUF_LENGTH: usize = MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH;
/// LTP history kept for 20 ms (`LTP_MEM_LENGTH_MS`).
const LTP_MEM_LENGTH_MS: i32 = 20;
/// Slack after the LTP buffers so a malformed lag never leaves them.
const LTP_SLACK: usize = 8;
/// `QUANT_LEVEL_ADJUST_Q10`.
const QUANT_LEVEL_ADJUST_Q10: i32 = 80;
/// `BWE_AFTER_LOSS_Q16`.
const BWE_AFTER_LOSS_Q16: i32 = 63570;
/// `N_LEVELS_QGAIN`.
const N_LEVELS_QGAIN: i32 = 64;
/// `MIN_DELTA_GAIN_QUANT`.
const MIN_DELTA_GAIN_QUANT: i32 = -4;
/// `MAX_DELTA_GAIN_QUANT`.
const MAX_DELTA_GAIN_QUANT: i32 = 36;
/// `OFFSET` in `gain_quant.c`: `MIN_QGAIN_DB * 128 / 6 + 16 * 128`.
const GAIN_OFFSET: i32 = (2 * 128) / 6 + 16 * 128;
/// `INV_SCALE_Q16` in `gain_quant.c`.
const GAIN_INV_SCALE_Q16: i32 = (65536 * (((88 - 2) * 128) / 6)) / (N_LEVELS_QGAIN - 1);

/// How a call decodes the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FrameMode {
    /// `FLAG_DECODE_NORMAL`.
    Normal,
    /// `FLAG_PACKET_LOST`: conceal.
    Lost,
    /// `FLAG_DECODE_LBRR`: decode the redundant copy, or conceal if absent.
    Lbrr,
}

/// Decoded parameters of one frame (`silk_decoder_control`).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DecoderControl {
    /// Pitch lag per subframe.
    pub pitch_l: [i32; MAX_NB_SUBFR],
    /// Gain per subframe in Q16.
    pub gains_q16: [i32; MAX_NB_SUBFR],
    /// LPC coefficients for the first and second half of the frame, Q12.
    pub pred_coef_q12: [[i16; MAX_LPC_ORDER]; 2],
    /// LTP taps per subframe, Q14.
    pub ltp_coef_q14: [i16; LTP_ORDER * MAX_NB_SUBFR],
    /// LTP state scaling, Q14.
    pub ltp_scale_q14: i32,
}

/// Decoder state of one channel (`silk_decoder_state`).
#[derive(Clone, Debug)]
pub(crate) struct ChannelState {
    pub prev_gain_q16: i32,
    pub exc_q14: [i32; MAX_FRAME_LENGTH],
    pub s_lpc_q14_buf: [i32; MAX_LPC_ORDER],
    pub out_buf: [i16; OUT_BUF_LENGTH],
    pub lag_prev: i32,
    pub last_gain_index: i32,
    pub fs_khz: i32,
    pub fs_api_hz: i32,
    pub nb_subfr: usize,
    pub frame_length: usize,
    pub subfr_length: usize,
    pub ltp_mem_length: usize,
    pub lpc_order: usize,
    pub prev_nlsf_q15: [i16; MAX_LPC_ORDER],
    pub first_frame_after_reset: bool,
    pub lag_low_bits_icdf: &'static [u8],
    pub contour_icdf: &'static [u8],
    pub n_frames_decoded: usize,
    pub n_frames_per_packet: usize,
    pub history: EntropyHistory,
    pub vad_flags: [bool; 3],
    pub lbrr_flag: bool,
    pub lbrr_flags: [bool; 3],
    pub resampler: Resampler,
    pub nlsf_cb: &'static NlsfCodebook,
    pub indices: SideInfo,
    pub cng: CngState,
    pub loss_cnt: i32,
    pub prev_signal_type: i32,
    pub plc: PlcState,
}

impl ChannelState {
    /// `silk_init_decoder`: a cleared channel that has not seen a frame.
    pub(crate) fn new() -> Self {
        let mut st = Self {
            prev_gain_q16: 65536,
            exc_q14: [0; MAX_FRAME_LENGTH],
            s_lpc_q14_buf: [0; MAX_LPC_ORDER],
            out_buf: [0; OUT_BUF_LENGTH],
            lag_prev: 0,
            last_gain_index: 0,
            fs_khz: 0,
            fs_api_hz: 0,
            nb_subfr: 0,
            frame_length: 0,
            subfr_length: 0,
            ltp_mem_length: 0,
            lpc_order: 0,
            prev_nlsf_q15: [0; MAX_LPC_ORDER],
            first_frame_after_reset: true,
            lag_low_bits_icdf: &UNIFORM4_ICDF,
            contour_icdf: &PITCH_CONTOUR_NB_ICDF,
            n_frames_decoded: 0,
            n_frames_per_packet: 0,
            history: EntropyHistory::default(),
            vad_flags: [false; 3],
            lbrr_flag: false,
            lbrr_flags: [false; 3],
            resampler: Resampler::default(),
            nlsf_cb: &NLSF_CB_NB_MB,
            indices: SideInfo::default(),
            cng: CngState::new(),
            loss_cnt: 0,
            prev_signal_type: 0,
            plc: PlcState::default(),
        };
        cng::reset(&mut st);
        plc::reset(&mut st);
        st
    }

    /// `silk_decoder_set_fs`: switches the internal rate (8, 12, or 16 kHz)
    /// and the frame size, and re-initializes the resampler when the
    /// internal or API rate changes.
    pub(crate) fn set_fs(&mut self, fs_khz: i32, fs_api_hz: i32) {
        self.subfr_length = 5 * fs_khz as usize;
        let frame_length = self.nb_subfr * self.subfr_length;
        if self.fs_khz != fs_khz || self.fs_api_hz != fs_api_hz {
            self.resampler = Resampler::new(fs_khz * 1000, fs_api_hz);
            self.fs_api_hz = fs_api_hz;
        }
        if self.fs_khz != fs_khz || frame_length != self.frame_length {
            let twenty_ms = self.nb_subfr == MAX_NB_SUBFR;
            self.contour_icdf = match (fs_khz == 8, twenty_ms) {
                (true, true) => &PITCH_CONTOUR_NB_ICDF,
                (true, false) => &PITCH_CONTOUR_10_MS_NB_ICDF,
                (false, true) => &PITCH_CONTOUR_ICDF,
                (false, false) => &PITCH_CONTOUR_10_MS_ICDF,
            };
            if self.fs_khz != fs_khz {
                self.ltp_mem_length = (LTP_MEM_LENGTH_MS * fs_khz) as usize;
                if fs_khz == 16 {
                    self.lpc_order = 16;
                    self.nlsf_cb = &NLSF_CB_WB;
                    self.lag_low_bits_icdf = &UNIFORM8_ICDF;
                } else {
                    self.lpc_order = 10;
                    self.nlsf_cb = &NLSF_CB_NB_MB;
                    self.lag_low_bits_icdf = if fs_khz == 12 {
                        &UNIFORM6_ICDF
                    } else {
                        &UNIFORM4_ICDF
                    };
                }
                self.first_frame_after_reset = true;
                self.lag_prev = 100;
                self.last_gain_index = 10;
                self.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
                self.out_buf = [0; OUT_BUF_LENGTH];
                self.s_lpc_q14_buf = [0; MAX_LPC_ORDER];
            }
            self.fs_khz = fs_khz;
            self.frame_length = frame_length;
        }
    }

    fn layout(&self) -> IndexLayout {
        IndexLayout {
            nb_subfr: self.nb_subfr,
            fs_khz: self.fs_khz,
            cb: self.nlsf_cb,
            lag_low_bits_icdf: self.lag_low_bits_icdf,
            contour_icdf: self.contour_icdf,
        }
    }

    /// `silk_decode_indices` plus `silk_decode_pulses` for an LBRR frame
    /// that normal decoding skips over.
    pub(crate) fn skip_frame(&mut self, rd: &mut RangeDecoder<'_>, cond: CondCoding) {
        let layout = self.layout();
        self.indices = decode_indices(rd, &layout, true, cond, &mut self.history);
        let mut pulses = [0i32; MAX_FRAME_LENGTH];
        decode_pulses(
            rd,
            &mut pulses,
            self.indices.signal_type,
            self.indices.quant_offset_type,
            self.frame_length,
        );
    }

    /// `silk_decode_frame`: decodes, conceals, or decodes the LBRR copy of
    /// one frame into `out[..frame_length]`, then runs PLC glue and CNG.
    pub(crate) fn decode_frame(
        &mut self,
        rd: &mut RangeDecoder<'_>,
        out: &mut [i16],
        mode: FrameMode,
        cond: CondCoding,
    ) -> usize {
        let mut ctrl = DecoderControl::default();
        let length = self.frame_length.min(out.len()).min(MAX_FRAME_LENGTH);
        let frame = &mut out[..length];
        let index = self.n_frames_decoded;
        let has_lbrr = self.lbrr_flags.get(index).copied().unwrap_or(false);
        if mode == FrameMode::Normal || (mode == FrameMode::Lbrr && has_lbrr) {
            let layout = self.layout();
            let active =
                mode == FrameMode::Lbrr || self.vad_flags.get(index).copied().unwrap_or(false);
            self.indices = decode_indices(rd, &layout, active, cond, &mut self.history);
            let mut pulses = [0i32; MAX_FRAME_LENGTH];
            decode_pulses(
                rd,
                &mut pulses,
                self.indices.signal_type,
                self.indices.quant_offset_type,
                self.frame_length,
            );
            self.decode_parameters(&mut ctrl, cond);
            self.decode_core(&mut ctrl, frame, &pulses);
            plc::update(self, &ctrl);
            self.loss_cnt = 0;
            self.prev_signal_type = self.indices.signal_type;
            self.first_frame_after_reset = false;
        } else {
            plc::conceal(self, &mut ctrl, frame);
        }

        // Keep the newest LTP memory of output.
        let mv_len = self.ltp_mem_length.saturating_sub(length);
        self.out_buf.copy_within(length..length + mv_len, 0);
        self.out_buf[mv_len..mv_len + length].copy_from_slice(frame);

        plc::glue_frames(self, frame);
        cng::apply(self, &ctrl, frame);

        self.lag_prev = ctrl.pitch_l[self.nb_subfr.clamp(1, MAX_NB_SUBFR) - 1];
        length
    }

    /// `silk_decode_parameters`: gains, LPC filters, pitch lags, and LTP
    /// filters from the frame's indices.
    fn decode_parameters(&mut self, ctrl: &mut DecoderControl, cond: CondCoding) {
        let order = self.lpc_order;
        gains_dequant(
            &mut ctrl.gains_q16,
            &self.indices.gains_indices,
            &mut self.last_gain_index,
            cond == CondCoding::Conditionally,
            self.nb_subfr,
        );

        let nlsf_q15 = nlsf::decode(self.nlsf_cb, &self.indices.nlsf_indices[..=order]);
        nlsf::nlsf2a(&mut ctrl.pred_coef_q12[1][..order], &nlsf_q15[..order]);
        // No interpolation right after a reset, such as a rate change.
        if self.first_frame_after_reset {
            self.indices.nlsf_interp_coef_q2 = 4;
        }
        let coef = self.indices.nlsf_interp_coef_q2;
        if coef < 4 {
            let mut nlsf0_q15 = [0i16; MAX_LPC_ORDER];
            for i in 0..order {
                let prev = i32::from(self.prev_nlsf_q15[i]);
                let diff = i32::from(nlsf_q15[i]) - prev;
                nlsf0_q15[i] = (prev + ((coef * diff) >> 2)) as i16;
            }
            nlsf::nlsf2a(&mut ctrl.pred_coef_q12[0][..order], &nlsf0_q15[..order]);
        } else {
            ctrl.pred_coef_q12[0] = ctrl.pred_coef_q12[1];
        }
        self.prev_nlsf_q15[..order].copy_from_slice(&nlsf_q15[..order]);

        // After a loss, chirp the filters.
        if self.loss_cnt != 0 {
            nlsf::bwexpander(&mut ctrl.pred_coef_q12[0][..order], BWE_AFTER_LOSS_Q16);
            nlsf::bwexpander(&mut ctrl.pred_coef_q12[1][..order], BWE_AFTER_LOSS_Q16);
        }

        if self.indices.signal_type == TYPE_VOICED {
            decode_pitch(
                self.indices.lag_index,
                self.indices.contour_index,
                &mut ctrl.pitch_l,
                self.fs_khz,
                self.nb_subfr,
            );
            let codebook: &[[i8; 5]] = match self.indices.per_index {
                0 => &LTP_GAIN_VQ_0,
                1 => &LTP_GAIN_VQ_1,
                _ => &LTP_GAIN_VQ_2,
            };
            for k in 0..self.nb_subfr {
                let taps = codebook
                    .get(self.indices.ltp_index[k])
                    .copied()
                    .unwrap_or([0; 5]);
                for (i, &t) in taps.iter().enumerate() {
                    ctrl.ltp_coef_q14[k * LTP_ORDER + i] = i16::from(t) << 7;
                }
            }
            ctrl.ltp_scale_q14 = i32::from(
                LTP_SCALES_Q14
                    .get(self.indices.ltp_scale_index)
                    .copied()
                    .unwrap_or(0),
            );
        } else {
            ctrl.pitch_l = [0; MAX_NB_SUBFR];
            ctrl.ltp_coef_q14 = [0; LTP_ORDER * MAX_NB_SUBFR];
            self.indices.per_index = 0;
            ctrl.ltp_scale_q14 = 0;
        }
    }

    /// `silk_decode_core`: excitation from the pulses, then the LTP and LPC
    /// synthesis filters per subframe.
    fn decode_core(&mut self, ctrl: &mut DecoderControl, xq: &mut [i16], pulses: &[i32]) {
        let st = &self.indices;
        let offset_q10 = QUANTIZATION_OFFSETS_Q10[usize::from(st.signal_type >= 2)]
            [usize::from(st.quant_offset_type != 0)];
        let nlsf_interpolation = st.nlsf_interp_coef_q2 < 4;
        let frame_length = self.frame_length.min(MAX_FRAME_LENGTH);
        let subfr_length = self.subfr_length.min(MAX_SUB_FRAME_LENGTH);
        let ltp_mem_length = self.ltp_mem_length;
        let order = self.lpc_order;

        // Excitation: pulses plus offset, signs scrambled by the seed.
        let mut seed = st.seed;
        for (i, &p) in pulses.iter().enumerate().take(frame_length) {
            seed = rand(seed);
            let mut e = lshift(p, 14);
            if e > 0 {
                e -= QUANT_LEVEL_ADJUST_Q10 << 4;
            } else if e < 0 {
                e += QUANT_LEVEL_ADJUST_Q10 << 4;
            }
            e = e.wrapping_add(offset_q10 << 4);
            if seed < 0 {
                e = e.wrapping_neg();
            }
            self.exc_q14[i] = e;
            seed = seed.wrapping_add(p);
        }

        let mut s_lpc_q14 = [0i32; MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER];
        s_lpc_q14[..MAX_LPC_ORDER].copy_from_slice(&self.s_lpc_q14_buf);
        let mut s_ltp = [0i16; MAX_FRAME_LENGTH];
        let mut s_ltp_q15 = [0i32; 2 * MAX_FRAME_LENGTH + LTP_SLACK];
        let mut res_q14 = [0i32; MAX_SUB_FRAME_LENGTH];
        let mut ltp_buf_idx = ltp_mem_length;
        // The largest lag whose rewhitening window stays in the buffer.
        let max_lag = ltp_mem_length.saturating_sub(order + LTP_ORDER / 2 + 1);

        for k in 0..self.nb_subfr.min(MAX_NB_SUBFR) {
            let a_q12 = ctrl.pred_coef_q12[k >> 1];
            let gain_q16 = ctrl.gains_q16[k];
            let gain_q10 = gain_q16 >> 6;
            let mut inv_gain_q31 = inverse32_varq(gain_q16, 47);
            let mut signal_type = self.indices.signal_type;

            // Rescale the LPC state when the gain changes.
            let gain_adj_q16 = if gain_q16 != self.prev_gain_q16 {
                let adj = div32_varq(self.prev_gain_q16, gain_q16, 16);
                for s in s_lpc_q14.iter_mut().take(MAX_LPC_ORDER) {
                    *s = smulww(adj, *s);
                }
                adj
            } else {
                1 << 16
            };
            self.prev_gain_q16 = gain_q16;

            // Avoid an abrupt switch from voiced concealment to unvoiced.
            if self.loss_cnt != 0
                && self.prev_signal_type == TYPE_VOICED
                && signal_type != TYPE_VOICED
                && k < MAX_NB_SUBFR / 2
            {
                let b = &mut ctrl.ltp_coef_q14[k * LTP_ORDER..(k + 1) * LTP_ORDER];
                b.fill(0);
                b[LTP_ORDER / 2] = 4096; // 0.25 in Q14
                signal_type = TYPE_VOICED;
                ctrl.pitch_l[k] = self.lag_prev;
            }
            let b_q14 = &ctrl.ltp_coef_q14[k * LTP_ORDER..(k + 1) * LTP_ORDER];

            if signal_type == TYPE_VOICED {
                let lag = usize::try_from(ctrl.pitch_l[k]).unwrap_or(0).min(max_lag);
                if k == 0 || (k == 2 && nlsf_interpolation) {
                    // Rewhiten the past output with this subframe's filter.
                    let start_idx = ltp_mem_length - lag - order - LTP_ORDER / 2;
                    if k == 2 {
                        let n = 2 * subfr_length;
                        self.out_buf[ltp_mem_length..ltp_mem_length + n].copy_from_slice(&xq[..n]);
                    }
                    let src = start_idx + k * subfr_length;
                    lpc_analysis_filter(
                        &mut s_ltp[start_idx..ltp_mem_length],
                        &self.out_buf[src..src + ltp_mem_length - start_idx],
                        &a_q12[..order],
                    );
                    if k == 0 {
                        // Scale down the LTP state to limit inter-packet
                        // dependency.
                        inv_gain_q31 = lshift(smulwb(inv_gain_q31, ctrl.ltp_scale_q14), 2);
                    }
                    for i in 0..lag + LTP_ORDER / 2 {
                        s_ltp_q15[ltp_buf_idx - i - 1] =
                            smulwb(inv_gain_q31, i32::from(s_ltp[ltp_mem_length - i - 1]));
                    }
                } else if gain_adj_q16 != 1 << 16 {
                    for i in 0..lag + LTP_ORDER / 2 {
                        let s = &mut s_ltp_q15[ltp_buf_idx - i - 1];
                        *s = smulww(gain_adj_q16, *s);
                    }
                }

                // Long-term prediction.
                let exc = &self.exc_q14[k * subfr_length..(k + 1) * subfr_length];
                for (res, &e) in res_q14.iter_mut().zip(exc) {
                    // The tap centred one lag back.
                    let pred = ltp_buf_idx + LTP_ORDER / 2 - lag;
                    let mut ltp_pred_q13: i32 = 2;
                    for (j, &b) in b_q14.iter().enumerate() {
                        ltp_pred_q13 = smlawb(ltp_pred_q13, s_ltp_q15[pred - j], i32::from(b));
                    }
                    *res = e.wrapping_add(lshift(ltp_pred_q13, 1));
                    s_ltp_q15[ltp_buf_idx] = lshift(*res, 1);
                    ltp_buf_idx += 1;
                }
            } else {
                let base = k * subfr_length;
                res_q14[..subfr_length].copy_from_slice(&self.exc_q14[base..base + subfr_length]);
            }

            // Short-term prediction and gain.
            let out = &mut xq[k * subfr_length..(k + 1) * subfr_length];
            for i in 0..subfr_length {
                let mut lpc_pred_q10 = (order >> 1) as i32;
                for (j, &a) in a_q12.iter().take(order).enumerate() {
                    lpc_pred_q10 = smlawb(
                        lpc_pred_q10,
                        s_lpc_q14[MAX_LPC_ORDER + i - j - 1],
                        i32::from(a),
                    );
                }
                let v = res_q14[i].wrapping_add(lshift(lpc_pred_q10, 4));
                s_lpc_q14[MAX_LPC_ORDER + i] = v;
                out[i] = sat16(rshift_round(smulww(v, gain_q10), 8));
            }
            s_lpc_q14.copy_within(subfr_length..subfr_length + MAX_LPC_ORDER, 0);
        }
        self.s_lpc_q14_buf
            .copy_from_slice(&s_lpc_q14[..MAX_LPC_ORDER]);
    }
}

/// `silk_gains_dequant`: gain indices to Q16 gains, tracking the previous
/// index across frames.
fn gains_dequant(
    gain_q16: &mut [i32; MAX_NB_SUBFR],
    ind: &[i32; MAX_NB_SUBFR],
    prev_ind: &mut i32,
    conditional: bool,
    nb_subfr: usize,
) {
    for k in 0..nb_subfr.min(MAX_NB_SUBFR) {
        if k == 0 && !conditional {
            // The index may not drop more than 16 steps (about 21.8 dB).
            *prev_ind = ind[k].max(*prev_ind - 16);
        } else {
            let ind_tmp = ind[k] + MIN_DELTA_GAIN_QUANT;
            let double_step_threshold = 2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN + *prev_ind;
            if ind_tmp > double_step_threshold {
                *prev_ind += (ind_tmp << 1) - double_step_threshold;
            } else {
                *prev_ind += ind_tmp;
            }
        }
        *prev_ind = (*prev_ind).clamp(0, N_LEVELS_QGAIN - 1);
        gain_q16[k] = log2lin((smulwb(GAIN_INV_SCALE_Q16, *prev_ind) + GAIN_OFFSET).min(3967));
    }
}

/// `silk_decode_pitch`: the pitch lag of each subframe from the lag index
/// and contour.
fn decode_pitch(
    lag_index: i16,
    contour_index: usize,
    pitch_lags: &mut [i32; MAX_NB_SUBFR],
    fs_khz: i32,
    nb_subfr: usize,
) {
    let min_lag = 2 * fs_khz;
    let max_lag = 18 * fs_khz;
    let lag = min_lag + i32::from(lag_index);
    for k in 0..nb_subfr.min(MAX_NB_SUBFR) {
        let offset = match (fs_khz == 8, nb_subfr == MAX_NB_SUBFR) {
            (true, true) => CB_LAGS_STAGE2[k].get(contour_index),
            (true, false) => CB_LAGS_STAGE2_10_MS
                .get(k)
                .and_then(|r| r.get(contour_index)),
            (false, true) => CB_LAGS_STAGE3[k].get(contour_index),
            (false, false) => CB_LAGS_STAGE3_10_MS
                .get(k)
                .and_then(|r| r.get(contour_index)),
        };
        let offset = i32::from(offset.copied().unwrap_or(0));
        pitch_lags[k] = (lag + offset).clamp(min_lag, max_lag);
    }
}

/// `silk_LPC_analysis_filter`: the prediction residual of `input` through
/// the Q12 filter `b`. The first `b.len()` outputs are zero.
pub(crate) fn lpc_analysis_filter(out: &mut [i16], input: &[i16], b: &[i16]) {
    let d = b.len();
    let len = out.len().min(input.len());
    for ix in d..len {
        let mut acc: i32 = 0;
        for (j, &c) in b.iter().enumerate() {
            acc = smlabb(acc, i32::from(input[ix - 1 - j]), i32::from(c));
        }
        let v = lshift(i32::from(input[ix]), 12).wrapping_sub(acc);
        out[ix] = sat16(rshift_round(v, 12));
    }
    let zero = d.min(out.len());
    out[..zero].fill(0);
}
