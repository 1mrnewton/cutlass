//! Band energy quantization (RFC 6716 section 4.3.2, `quant_bands.c`).
//!
//! Energies are base-2 logarithms of the band amplitude relative to
//! `E_MEANS`, one value per band and channel, stored channel-major with
//! `NB_EBANDS` per channel.

use std::ops::Range;

use super::laplace;
use super::rate::Allocation;
use super::tables::{
    BETA_COEF, BETA_INTRA, E_MEANS, E_PROB_MODEL, MAX_FINE_BITS, NB_EBANDS, PRED_COEF,
    SMALL_ENERGY_ICDF,
};
use crate::range::{RangeDecoder, RangeEncoder};

/// Decodes the coarse energy (`unquant_coarse_energy`).
pub(crate) fn unquant_coarse(
    old: &mut [f32],
    start: usize,
    end: usize,
    intra: bool,
    dec: &mut RangeDecoder<'_>,
    channels: usize,
    lm: usize,
) {
    let model = &E_PROB_MODEL[lm][usize::from(intra)];
    let (coef, beta) = if intra {
        (0.0, BETA_INTRA)
    } else {
        (PRED_COEF[lm], BETA_COEF[lm])
    };
    let budget = dec.storage() as i32 * 8;
    let mut prev = [0f32; 2];
    for i in start..end {
        for (c, prev_c) in prev.iter_mut().enumerate().take(channels) {
            let tell = dec.tell();
            let qi: i32 = if budget - tell >= 15 {
                let pi = 2 * i.min(20);
                laplace::decode(
                    dec,
                    u32::from(model[pi]) << 7,
                    u32::from(model[pi + 1]) << 6,
                )
            } else if budget - tell >= 2 {
                let q = dec.dec_icdf(&SMALL_ENERGY_ICDF, 2) as i32;
                (q >> 1) ^ -(q & 1)
            } else if budget - tell >= 1 {
                -i32::from(dec.bit_logp(1))
            } else {
                -1
            };
            let q = qi as f32;
            let slot = &mut old[i + c * NB_EBANDS];
            *slot = slot.max(-9.0);
            let tmp = coef * *slot + *prev_c + q;
            *slot = tmp;
            *prev_c = *prev_c + q - beta * q;
        }
    }
}

/// Decodes the fine energy bits (`unquant_fine_energy`).
pub(crate) fn unquant_fine(
    old: &mut [f32],
    start: usize,
    end: usize,
    fine_quant: &[i32; NB_EBANDS],
    dec: &mut RangeDecoder<'_>,
    channels: usize,
) {
    for i in start..end {
        let bits = fine_quant[i];
        if bits <= 0 {
            continue;
        }
        for c in 0..channels {
            let q2 = dec.dec_bits(bits as u32);
            let offset = (q2 as f32 + 0.5) * (1 << (14 - bits)) as f32 * (1.0 / 16384.0) - 0.5;
            old[i + c * NB_EBANDS] += offset;
        }
    }
}

/// Spends the bits left after PVQ on one more energy bit per band
/// (`unquant_energy_finalise`).
pub(crate) fn unquant_final(
    old: &mut [f32],
    bands: Range<usize>,
    alloc: &Allocation,
    mut bits_left: i32,
    dec: &mut RangeDecoder<'_>,
    channels: usize,
) {
    let (start, end) = (bands.start, bands.end);
    let fine_quant = &alloc.fine_quant;
    let fine_priority = &alloc.fine_priority;
    let c_bits = channels as i32;
    for prio in 0..2 {
        let mut i = start;
        while i < end && bits_left >= c_bits {
            if fine_quant[i] >= MAX_FINE_BITS || fine_priority[i] != prio {
                i += 1;
                continue;
            }
            for c in 0..channels {
                let q2 = dec.dec_bits(1);
                let offset =
                    (q2 as f32 - 0.5) * (1 << (14 - fine_quant[i] - 1)) as f32 * (1.0 / 16384.0);
                old[i + c * NB_EBANDS] += offset;
                bits_left -= 1;
            }
            i += 1;
        }
    }
}

/// Linear band amplitudes from log energies (`log2Amp`), in the 32768
/// signal scale, with the RFC 8251 section 8 cap.
pub(crate) fn log2amp(band_e: &mut [f32], old: &[f32], start: usize, end: usize, channels: usize) {
    for c in 0..channels {
        for (i, &mean) in E_MEANS.iter().enumerate().take(NB_EBANDS) {
            let at = i + c * NB_EBANDS;
            band_e[at] = if i < start || i >= end {
                0.0
            } else {
                exp2((old[at] + mean).min(32.0))
            };
        }
    }
}

/// Log energies relative to the means from amplitudes (`amp2Log2`).
pub(crate) fn amp2log(
    band_log_e: &mut [f32],
    band_e: &[f32],
    eff_end: usize,
    end: usize,
    channels: usize,
) {
    for c in 0..channels {
        for (i, &mean) in E_MEANS.iter().enumerate().take(eff_end) {
            let at = i + c * NB_EBANDS;
            band_log_e[at] = log2(band_e[at]) - mean;
        }
        for i in eff_end..end {
            band_log_e[i + c * NB_EBANDS] = -14.0;
        }
    }
}

#[inline]
pub(crate) fn exp2(x: f32) -> f32 {
    (std::f64::consts::LN_2 * f64::from(x)).exp() as f32
}

#[inline]
pub(crate) fn log2(x: f32) -> f32 {
    (std::f64::consts::LOG2_E * f64::from(x).ln()) as f32
}

/// The fixed inputs of one coarse energy pass.
pub(crate) struct CoarseParams<'p> {
    /// The band energies to code.
    pub e_bands: &'p [f32],
    pub start: usize,
    pub end: usize,
    /// Frame size in bits.
    pub budget: i32,
    /// Largest drop in energy per frame the coder follows exactly.
    pub max_decay: f32,
    pub channels: usize,
    pub lm: usize,
}

/// Encoder side of coarse energy (`quant_coarse_energy_impl`); returns the
/// badness (how far the coded values were clamped).
pub(crate) fn quant_coarse_impl(
    p: &CoarseParams<'_>,
    old: &mut [f32],
    error: &mut [f32],
    intra: bool,
    enc: &mut RangeEncoder,
) -> i32 {
    let (e_bands, start, end, budget, max_decay, channels, lm) = (
        p.e_bands,
        p.start,
        p.end,
        p.budget,
        p.max_decay,
        p.channels,
        p.lm,
    );
    let tell = enc.tell();
    if tell + 3 <= budget {
        enc.bit_logp(intra, 3);
    }
    let model = &E_PROB_MODEL[lm][usize::from(intra)];
    let (coef, beta) = if intra {
        (0.0, BETA_INTRA)
    } else {
        (PRED_COEF[lm], BETA_COEF[lm])
    };
    let mut prev = [0f32; 2];
    let mut badness = 0;
    for i in start..end {
        for (c, prev_c) in prev.iter_mut().enumerate().take(channels) {
            let at = i + c * NB_EBANDS;
            let x = e_bands[at];
            let old_e = old[at].max(-9.0);
            let f = x - coef * old_e - *prev_c;
            let mut qi = (0.5 + f).floor() as i32;
            let decay_bound = old[at].max(-28.0) - max_decay;
            if qi < 0 && x < decay_bound {
                qi += (decay_bound - x) as i32;
                if qi > 0 {
                    qi = 0;
                }
            }
            let qi0 = qi;
            let tell = enc.tell();
            let bits_left = budget - tell - 3 * channels as i32 * (end - i) as i32;
            if i != start && bits_left < 30 {
                if bits_left < 24 {
                    qi = qi.min(1);
                }
                if bits_left < 16 {
                    qi = qi.max(-1);
                }
            }
            if budget - tell >= 15 {
                let pi = 2 * i.min(20);
                qi = laplace::encode(
                    enc,
                    qi,
                    u32::from(model[pi]) << 7,
                    u32::from(model[pi + 1]) << 6,
                );
            } else if budget - tell >= 2 {
                qi = qi.clamp(-1, 1);
                enc.enc_icdf(
                    ((2 * qi) ^ -i32::from(qi < 0)) as usize,
                    &SMALL_ENERGY_ICDF,
                    2,
                );
            } else if budget - tell >= 1 {
                qi = qi.min(0);
                enc.bit_logp(qi != 0, 1);
            } else {
                qi = -1;
            }
            error[at] = f - qi as f32;
            badness += (qi0 - qi).abs();
            let q = qi as f32;
            old[at] = coef * old_e + *prev_c + q;
            *prev_c = *prev_c + q - beta * q;
        }
    }
    badness
}

/// Encoder side of fine energy (`quant_fine_energy`).
pub(crate) fn quant_fine(
    old: &mut [f32],
    error: &mut [f32],
    start: usize,
    end: usize,
    fine_quant: &[i32; NB_EBANDS],
    enc: &mut RangeEncoder,
    channels: usize,
) {
    for (i, &bits) in fine_quant.iter().enumerate().take(end).skip(start) {
        if bits <= 0 {
            continue;
        }
        let frac = 1i32 << bits;
        for c in 0..channels {
            let at = i + c * NB_EBANDS;
            let q2 = (((error[at] + 0.5) * frac as f32).floor() as i32).clamp(0, frac - 1);
            enc.enc_bits(q2 as u32, bits as u32);
            let offset = (q2 as f32 + 0.5) * (1 << (14 - bits)) as f32 * (1.0 / 16384.0) - 0.5;
            old[at] += offset;
            error[at] -= offset;
        }
    }
}

/// Encoder side of the final energy bits (`quant_energy_finalise`).
pub(crate) fn quant_final(
    old: &mut [f32],
    error: &mut [f32],
    bands: Range<usize>,
    alloc: &Allocation,
    mut bits_left: i32,
    enc: &mut RangeEncoder,
    channels: usize,
) {
    let (start, end) = (bands.start, bands.end);
    let fine_quant = &alloc.fine_quant;
    let fine_priority = &alloc.fine_priority;
    let c_bits = channels as i32;
    for prio in 0..2 {
        let mut i = start;
        while i < end && bits_left >= c_bits {
            if fine_quant[i] >= MAX_FINE_BITS || fine_priority[i] != prio {
                i += 1;
                continue;
            }
            for c in 0..channels {
                let at = i + c * NB_EBANDS;
                let q2 = u32::from(error[at] >= 0.0);
                enc.enc_bits(q2, 1);
                let offset =
                    (q2 as f32 - 0.5) * (1 << (14 - fine_quant[i] - 1)) as f32 * (1.0 / 16384.0);
                old[at] += offset;
                error[at] -= offset;
                bits_left -= 1;
            }
            i += 1;
        }
    }
}
