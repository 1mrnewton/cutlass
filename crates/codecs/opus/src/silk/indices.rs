//! Side information of one SILK frame: signal type, gains, NLSF indices,
//! pitch lags, LTP filters, and the excitation seed (RFC 6716 sections
//! 4.2.7.3-4.2.7.7; reference `decode_indices.c`).

use super::nlsf::{MAX_LPC_ORDER, NLSF_QUANT_MAX_AMPLITUDE, unpack};
use super::tables::{
    DELTA_GAIN_ICDF, GAIN_ICDF, LTP_GAIN_ICDF, LTP_PER_INDEX_ICDF, LTP_SCALE_ICDF, NLSF_EXT_ICDF,
    NLSF_INTERPOLATION_FACTOR_ICDF, NlsfCodebook, PITCH_DELTA_ICDF, PITCH_LAG_ICDF,
    TYPE_OFFSET_NO_VAD_ICDF, TYPE_OFFSET_VAD_ICDF, UNIFORM4_ICDF, UNIFORM8_ICDF,
};
use crate::range::RangeDecoder;

/// Subframes in a 20 ms frame.
pub(crate) const MAX_NB_SUBFR: usize = 4;

/// `TYPE_NO_VOICE_ACTIVITY`.
pub(crate) const TYPE_NO_VOICE_ACTIVITY: i32 = 0;
/// `TYPE_VOICED`.
pub(crate) const TYPE_VOICED: i32 = 2;

/// How the frame's first gain and LTP scaling are coded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CondCoding {
    /// `CODE_INDEPENDENTLY`: absolute first gain, LTP scaling present.
    Independently,
    /// `CODE_INDEPENDENTLY_NO_LTP_SCALING`: absolute first gain, no LTP
    /// scaling (the side channel after a mid-only frame).
    IndependentlyNoLtpScaling,
    /// `CODE_CONDITIONALLY`: the first gain is a delta.
    Conditionally,
}

/// The quantization indices of one frame (`SideInfoIndices`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SideInfo {
    /// 0 inactive, 1 unvoiced, 2 voiced.
    pub signal_type: i32,
    /// 0 low, 1 high quantization offset.
    pub quant_offset_type: i32,
    /// Gain indices per subframe: absolute or delta for the first.
    pub gains_indices: [i32; MAX_NB_SUBFR],
    /// First-stage NLSF index, then the residual index per coefficient.
    pub nlsf_indices: [i32; MAX_LPC_ORDER + 1],
    /// NLSF interpolation weight in Q2; 4 means none.
    pub nlsf_interp_coef_q2: i32,
    /// Primary pitch lag index.
    pub lag_index: i16,
    /// Pitch contour index.
    pub contour_index: usize,
    /// LTP periodicity (codebook) index.
    pub per_index: usize,
    /// LTP filter index per subframe.
    pub ltp_index: [usize; MAX_NB_SUBFR],
    /// LTP scaling index.
    pub ltp_scale_index: usize,
    /// Excitation seed, 0 to 3.
    pub seed: i32,
}

/// The frame layout that index decoding depends on.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IndexLayout {
    /// Subframes, 2 or 4.
    pub nb_subfr: usize,
    /// Internal rate in kHz.
    pub fs_khz: i32,
    /// NLSF codebook for the rate.
    pub cb: &'static NlsfCodebook,
    /// Low bits of the absolute pitch lag.
    pub lag_low_bits_icdf: &'static [u8],
    /// Pitch contour table for the rate and frame size.
    pub contour_icdf: &'static [u8],
}

/// Coding history the conditional pitch lag needs.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct EntropyHistory {
    /// `ec_prevSignalType`.
    pub prev_signal_type: i32,
    /// `ec_prevLagIndex`.
    pub prev_lag_index: i16,
}

/// A sub-table of `table` starting at `start`, or an always-zero table if
/// the offset is out of range (it never is for the reference tables).
pub(crate) fn sub_icdf(table: &[u8], start: usize) -> &[u8] {
    table.get(start..).unwrap_or(&[0])
}

/// `silk_decode_indices`. `active` is the frame's VAD flag, or true for an
/// LBRR frame.
pub(crate) fn decode_indices(
    rd: &mut RangeDecoder<'_>,
    layout: &IndexLayout,
    active: bool,
    cond: CondCoding,
    hist: &mut EntropyHistory,
) -> SideInfo {
    let mut ix = SideInfo::default();

    // Signal type and quantizer offset.
    let type_offset = if active {
        rd.dec_icdf(&TYPE_OFFSET_VAD_ICDF, 8) + 2
    } else {
        rd.dec_icdf(&TYPE_OFFSET_NO_VAD_ICDF, 8)
    };
    ix.signal_type = (type_offset >> 1) as i32;
    ix.quant_offset_type = (type_offset & 1) as i32;

    // Gains: the first absolute (3 MSBs by signal type, 3 LSBs uniform) or
    // a delta, the rest deltas.
    if cond == CondCoding::Conditionally {
        ix.gains_indices[0] = rd.dec_icdf(&DELTA_GAIN_ICDF, 8) as i32;
    } else {
        let msb = rd.dec_icdf(&GAIN_ICDF[type_offset >> 1], 8) as i32;
        ix.gains_indices[0] = (msb << 3) + rd.dec_icdf(&UNIFORM8_ICDF, 8) as i32;
    }
    for g in ix.gains_indices.iter_mut().take(layout.nb_subfr).skip(1) {
        *g = rd.dec_icdf(&DELTA_GAIN_ICDF, 8) as i32;
    }

    // NLSF indices.
    let cb = layout.cb;
    let start = (type_offset >> 2) * cb.n_vectors;
    let cb1_icdf = cb.cb1_icdf.get(start..start + cb.n_vectors).unwrap_or(&[0]);
    let cb1 = rd.dec_icdf(cb1_icdf, 8);
    ix.nlsf_indices[0] = cb1 as i32;
    let (ec_ix, _) = unpack(cb, cb1);
    let max_amp = NLSF_QUANT_MAX_AMPLITUDE;
    for (i, &ec) in ec_ix.iter().enumerate().take(cb.order) {
        let mut v = rd.dec_icdf(sub_icdf(cb.ec_icdf, ec), 8) as i32;
        if v == 0 {
            v -= rd.dec_icdf(&NLSF_EXT_ICDF, 8) as i32;
        } else if v == 2 * max_amp {
            v += rd.dec_icdf(&NLSF_EXT_ICDF, 8) as i32;
        }
        ix.nlsf_indices[i + 1] = v - max_amp;
    }
    ix.nlsf_interp_coef_q2 = if layout.nb_subfr == MAX_NB_SUBFR {
        rd.dec_icdf(&NLSF_INTERPOLATION_FACTOR_ICDF, 8) as i32
    } else {
        4
    };

    if ix.signal_type == TYPE_VOICED {
        // Pitch lag: a delta from the previous frame's when allowed,
        // otherwise absolute.
        let mut absolute = true;
        if cond == CondCoding::Conditionally && hist.prev_signal_type == TYPE_VOICED {
            let delta = rd.dec_icdf(&PITCH_DELTA_ICDF, 8) as i16;
            if delta > 0 {
                ix.lag_index = hist.prev_lag_index.wrapping_add(delta - 9);
                absolute = false;
            }
        }
        if absolute {
            let high = rd.dec_icdf(&PITCH_LAG_ICDF, 8) as i32 * (layout.fs_khz >> 1);
            let low = rd.dec_icdf(layout.lag_low_bits_icdf, 8) as i32;
            ix.lag_index = (high + low) as i16;
        }
        hist.prev_lag_index = ix.lag_index;
        ix.contour_index = rd.dec_icdf(layout.contour_icdf, 8);

        // LTP filters.
        ix.per_index = rd.dec_icdf(&LTP_PER_INDEX_ICDF, 8);
        let gain_icdf = LTP_GAIN_ICDF.get(ix.per_index).copied().unwrap_or(&[0]);
        for index in &mut ix.ltp_index[..layout.nb_subfr] {
            *index = rd.dec_icdf(gain_icdf, 8);
        }
        ix.ltp_scale_index = if cond == CondCoding::Independently {
            rd.dec_icdf(&LTP_SCALE_ICDF, 8)
        } else {
            0
        };
    }
    hist.prev_signal_type = ix.signal_type;

    ix.seed = rd.dec_icdf(&UNIFORM4_ICDF, 8) as i32;
    ix
}
