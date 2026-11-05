//! The Laplace-like model of coarse energy deltas (RFC 6716 section
//! 4.3.2.1, `laplace.c`).
//!
//! `fs` is the probability of zero out of 32768 and `decay` the ratio of
//! consecutive magnitudes in Q14. Every value has at least probability
//! 1/32768, so any integer the encoder clamps to the representable range
//! codes.

use crate::range::{RangeDecoder, RangeEncoder};

/// The smallest probability of one value (`LAPLACE_MINP`).
const MINP: u32 = 1;
/// Values guaranteed representable in each direction (`LAPLACE_NMIN`).
const NMIN: u32 = 16;

/// Probability of +1 (and of -1).
fn freq1(fs0: u32, decay: u32) -> u32 {
    let ft = 32768u32.saturating_sub(MINP * (2 * NMIN) + fs0);
    (ft * (16384u32.saturating_sub(decay))) >> 15
}

/// Codes `value`, clamped to what the model can represent; returns the
/// value actually coded.
pub(crate) fn encode(enc: &mut RangeEncoder, value: i32, fs: u32, decay: u32) -> i32 {
    let mut coded = value;
    let mut fl = 0u32;
    let mut fs = fs;
    if value != 0 {
        let s: i32 = -i32::from(value < 0);
        let val = ((value + s) ^ s) as u32;
        fl = fs;
        fs = freq1(fs, decay);
        let mut i = 1u32;
        while fs > 0 && i < val {
            fs *= 2;
            fl += fs + 2 * MINP;
            fs = (fs * decay) >> 15;
            i += 1;
        }
        if fs == 0 {
            let mut ndi_max = (32768u32.saturating_sub(fl) + MINP - 1) as i32;
            ndi_max = (ndi_max - s) >> 1;
            let di = (val as i32 - i as i32).min(ndi_max - 1);
            fl = (fl as i32 + (2 * di + 1 + s) * MINP as i32) as u32;
            fs = MINP.min(32768u32.saturating_sub(fl));
            coded = (i as i32 + di + s) ^ s;
        } else {
            fs += MINP;
            fl += fs & !(s as u32);
        }
    }
    enc.encode_bin(fl, fl + fs, 15);
    coded
}

pub(crate) fn decode(dec: &mut RangeDecoder<'_>, fs: u32, decay: u32) -> i32 {
    let mut val = 0i32;
    let fm = dec.decode_bin(15);
    let mut fl = 0u32;
    let mut fs = fs;
    if fm >= fs {
        val += 1;
        fl = fs;
        fs = freq1(fs, decay) + MINP;
        while fs > MINP && fm >= fl + 2 * fs {
            fs *= 2;
            fl += fs;
            fs = ((fs - 2 * MINP) * decay) >> 15;
            fs += MINP;
            val += 1;
        }
        if fs <= MINP {
            let di = (fm - fl) >> 1;
            val += di as i32;
            fl += 2 * di * MINP;
        }
        if fm < fl + fs {
            val = -val;
        } else {
            fl += fs;
        }
    }
    let fh = (fl + fs).min(32768);
    let fl = fl.min(fh.saturating_sub(1));
    dec.update(fl, fh, 32768);
    val
}

#[cfg(test)]
mod tests {
    use super::{decode, encode};
    use crate::range::{RangeDecoder, RangeEncoder};

    #[test]
    fn every_value_round_trips_for_every_model() {
        // The e_prob_model extremes: a likely zero with slow decay, an
        // unlikely zero with fast decay.
        for &(fs, decay) in &[
            (72u32 << 7, 127u32 << 6),
            (22 << 7, 178 << 6),
            (190 << 7, 6 << 6),
        ] {
            let values: Vec<i32> = (-40..=40).chain([-1000, 1000, 30000, -30000]).collect();
            let mut enc = RangeEncoder::new(4000);
            let coded: Vec<i32> = values
                .iter()
                .map(|&v| encode(&mut enc, v, fs, decay))
                .collect();
            enc.done();
            assert!(!enc.error());
            let bytes = enc.into_bytes();
            let mut dec = RangeDecoder::new(&bytes);
            for (i, &want) in coded.iter().enumerate() {
                assert_eq!(decode(&mut dec, fs, decay), want, "value {}", values[i]);
            }
            for (v, c) in values.iter().zip(&coded) {
                if v.abs() <= 16 {
                    assert_eq!(v, c, "small values code exactly");
                }
            }
        }
    }

    #[test]
    fn decoding_random_bytes_never_panics() {
        let mut seed = 1u32;
        for _ in 0..200 {
            let bytes: Vec<u8> = (0..16)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 24) as u8
                })
                .collect();
            let mut dec = RangeDecoder::new(&bytes);
            for _ in 0..40 {
                let v = decode(&mut dec, 6 << 7, 200 << 6);
                assert!(v.abs() < 40000);
            }
        }
    }
}
