//! Residue values from a spectrum and its coded floor: the spectrum over
//! the floor, rounded, with noise normalization, and square-polar
//! coupling of channel pairs (Vorbis I section 1.3.3, inverted in section
//! 4.3.5).
//!
//! Below the point-stereo bin a pair is coupled losslessly: the decoder
//! gets both rounded channels back exactly. Above it, both channels share
//! one magnitude that keeps their summed energy, and the angle is 0.

use crate::encoder::profile::MAX_RESIDUE;

/// Noise normalization: in each run of `width` bins from `start`, when
/// the energy rounded away is at least `threshold`, the largest value
/// that rounded to 0 becomes one pulse that carries that energy.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Normalize {
    pub start: usize,
    pub width: usize,
    pub threshold: f32,
}

/// Rounds `ratio` (spectrum over floor) into `out`, normalizing from
/// `normalize.start`. `bands` are absolute bin edges; `origin` is the bin
/// `ratio[0]` represents. Empty `bands` uses `normalize.width` instead.
/// Bins from `ratio.len()` up are 0.
pub(crate) fn quantize(
    ratio: &[f32],
    normalize: &Normalize,
    bands: &[usize],
    origin: usize,
    out: &mut [i32],
) {
    out.fill(0);
    // With bands, ±1 is rounded away and one pulse per band carries that
    // energy. A grid of ±1 is what the residue books spend bits on.
    let dead_zone = bands.len() >= 2;
    for (slot, &r) in out.iter_mut().zip(ratio) {
        let rounded = (r.round() as i32).clamp(-MAX_RESIDUE, MAX_RESIDUE);
        *slot = if dead_zone && rounded.abs() <= 2 {
            0
        } else {
            rounded
        };
    }
    if bands.len() >= 2 {
        let mut band = bands.partition_point(|&edge| edge <= origin);
        band = band.saturating_sub(1);
        let first = normalize.start.saturating_sub(origin);
        while band + 1 < bands.len() {
            let start = bands[band].saturating_sub(origin).max(first);
            let end = bands[band + 1].saturating_sub(origin);
            // A half-Bark above a few kilohertz is wider than one pulse can
            // carry. Split it so each piece keeps its energy.
            let piece = normalize.width.max(8);
            let mut at = start;
            while at < end {
                let stop = (at + piece).min(end);
                restore_window(ratio, out, at, stop, normalize.threshold);
                at = stop;
            }
            band += 1;
        }
        return;
    }
    let width = normalize.width.max(1);
    let mut start = normalize.start.saturating_sub(origin).min(ratio.len());
    while start < ratio.len() {
        let end = (start + width).min(ratio.len());
        restore_window(ratio, out, start, end, normalize.threshold);
        start = end;
    }
}

/// One pulse at the largest discarded bin, with amplitude `sqrt(energy)`,
/// when the discarded energy in `start..end` reaches `threshold`.
fn restore_window(ratio: &[f32], out: &mut [i32], start: usize, end: usize, threshold: f32) {
    let end = end.min(ratio.len());
    if start >= end {
        return;
    }
    let mut energy = 0.0_f32;
    let mut best = start;
    let mut best_abs = -1.0_f32;
    for k in start..end {
        if out[k] == 0 {
            energy += ratio[k] * ratio[k];
            let magnitude = ratio[k].abs();
            if magnitude > best_abs {
                best_abs = magnitude;
                best = k;
            }
        }
    }
    if energy >= threshold && best_abs >= 0.0 {
        // Stay in a small residue class. A bigger pulse codes the whole
        // partition with an expensive book.
        let amp = (energy.sqrt().round() as i32).clamp(1, 4);
        out[best] = if ratio[best] < 0.0 { -amp } else { amp };
    }
}

/// Lossless square-polar coupling of rounded `left` and `right`: the
/// magnitude and angle the decoder turns back into them.
pub(crate) fn couple(left: i32, right: i32) -> (i32, i32) {
    if left.abs() > right.abs() {
        if left > 0 {
            (left, left - right)
        } else {
            (left, right - left)
        }
    } else if right > 0 {
        (right, left - right)
    } else {
        (right, right - left)
    }
}

/// The decoder's inverse (section 4.3.5).
#[cfg(test)]
pub(crate) fn uncouple(magnitude: i32, angle: i32) -> (i32, i32) {
    if magnitude > 0 {
        if angle > 0 {
            (magnitude, magnitude - angle)
        } else {
            (magnitude + angle, magnitude)
        }
    } else if angle > 0 {
        (magnitude, magnitude + angle)
    } else {
        (magnitude - angle, magnitude)
    }
}

/// One channel pair's spectra and coded floors.
pub(crate) struct Pair<'a> {
    pub spectra: [&'a [f32]; 2],
    pub floors: [&'a [f32]; 2],
    /// A channel whose floor is unused decodes to silence.
    pub silent: [bool; 2],
}

/// Residue values of a coupled pair, interleaved magnitude and angle, for
/// the first `bins` bins; point stereo from `point`.
pub(crate) fn quantize_pair(
    pair: &Pair<'_>,
    bins: usize,
    point: usize,
    normalize: &Normalize,
    bands: &[usize],
    scratch: &mut PairScratch,
    out: &mut [i32],
) {
    out.fill(0);
    let point = if pair.silent[0] || pair.silent[1] {
        bins
    } else {
        point.min(bins)
    };
    let ratio = |c: usize, k: usize| -> f32 {
        if pair.silent[c] {
            0.0
        } else {
            pair.spectra[c][k] / pair.floors[c][k]
        }
    };
    for c in 0..2 {
        scratch.ratio[c].clear();
        scratch.ratio[c].extend((0..point).map(|k| ratio(c, k)));
        scratch.values[c].resize(point, 0);
        quantize(
            &scratch.ratio[c],
            normalize,
            bands,
            0,
            &mut scratch.values[c],
        );
    }
    for k in 0..point {
        let (m, a) = couple(scratch.values[0][k], scratch.values[1][k]);
        out[2 * k] = m.clamp(-MAX_RESIDUE, MAX_RESIDUE);
        out[2 * k + 1] = a.clamp(-MAX_RESIDUE, MAX_RESIDUE);
    }
    if point < bins {
        scratch.ratio[0].clear();
        for k in point..bins {
            let (l, r) = (pair.spectra[0][k], pair.spectra[1][k]);
            let (fl, fr) = (pair.floors[0][k], pair.floors[1][k]);
            let magnitude = ((l * l + r * r) / (fl * fl + fr * fr)).sqrt();
            let sign = if l + r < 0.0 { -1.0 } else { 1.0 };
            scratch.ratio[0].push(sign * magnitude);
        }
        scratch.values[0].resize(bins - point, 0);
        quantize(
            &scratch.ratio[0],
            normalize,
            bands,
            point,
            &mut scratch.values[0],
        );
        for k in point..bins {
            out[2 * k] = scratch.values[0][k - point];
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PairScratch {
    ratio: [Vec<f32>; 2],
    values: [Vec<i32>; 2],
}

#[cfg(test)]
mod tests {
    use super::{Normalize, Pair, PairScratch, couple, quantize, quantize_pair, uncouple};

    #[test]
    fn coupling_inverts_through_the_decoder() {
        for left in -40..=40 {
            for right in -40..=40 {
                let (m, a) = couple(left, right);
                assert_eq!(uncouple(m, a), (left, right), "{left} {right}");
                assert!(a.abs() <= 2 * m.abs());
            }
        }
    }

    #[test]
    fn normalization_fills_energy_that_rounding_dropped() {
        let off = Normalize {
            start: usize::MAX,
            width: 16,
            threshold: 1.0,
        };
        let on = Normalize { start: 0, ..off };
        let ratio = [0.45_f32, -0.44, 0.43, 0.42, 0.1, 2.6, 0.0, 0.0];
        let mut plain = [0; 8];
        quantize(&ratio, &off, &[], 0, &mut plain);
        assert_eq!(plain, [0, 0, 0, 0, 0, 3, 0, 0]);
        let mut filled = [0; 8];
        quantize(&ratio, &on, &[], 0, &mut filled);
        // 0.765 of energy rounded away: below the threshold, nothing fills.
        assert_eq!(filled, plain);
        let loud = [0.49_f32, -0.48, 0.47, 0.46, 0.45, 0.44, 0.1, 0.0];
        quantize(&loud, &on, &[], 0, &mut filled);
        assert_eq!(filled, [1, 0, 0, 0, 0, 0, 0, 0]);
        // 1.31 dropped, and one pulse of amplitude 1 carries it.
        let quiet = Normalize {
            threshold: 0.25,
            ..on
        };
        quantize(&loud, &quiet, &[], 0, &mut filled);
        assert_eq!(filled, [1, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn point_stereo_keeps_the_pair_energy() {
        let left = [3.0_f32, -2.0, 8.0, 0.0];
        let right = [1.0_f32, -2.0, -4.0, 5.0];
        let floors = [1.0_f32; 4];
        let pair = Pair {
            spectra: [&left, &right],
            floors: [&floors, &floors],
            silent: [false, false],
        };
        let normalize = Normalize {
            start: usize::MAX,
            width: 16,
            threshold: 1.0,
        };
        let mut out = [0; 8];
        quantize_pair(
            &pair,
            4,
            2,
            &normalize,
            &[],
            &mut PairScratch::default(),
            &mut out,
        );
        assert_eq!(uncouple(out[0], out[1]), (3, 1));
        assert_eq!(uncouple(out[2], out[3]), (-2, -2));
        // sqrt((64 + 16) / 2) = 6.3, the sign of the sum.
        assert_eq!((out[4], out[5]), (6, 0));
        assert_eq!((out[6], out[7]), (4, 0));
        let silent = Pair {
            silent: [false, true],
            ..pair
        };
        quantize_pair(
            &silent,
            4,
            2,
            &normalize,
            &[],
            &mut PairScratch::default(),
            &mut out,
        );
        assert_eq!(uncouple(out[4], out[5]), (8, 0));
    }
}
