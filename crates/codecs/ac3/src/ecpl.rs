//! Enhanced coupling (Annex E 2.3.3.14 and 3.5).
//!
//! The coupling channel is turned back into a time signal with the
//! neighboring blocks, so a phase rotation is not applied to time-aliased
//! coefficients, then each channel is scaled and rotated back into the
//! MDCT domain. A/52 does not name the random sequence used when the chaos
//! value is nonzero. This decoder draws it from the same linear congruential
//! generator as dither, so a chaos of zero does not depend on that choice.

use std::f32::consts::PI;

use crate::frame::MAX_FBW;
use crate::mdct::{COEFS, SPAN, Transform};

/// First transform coefficient of each enhanced-coupling sub-band, plus the
/// sentinel after the last one. Table E3.9.
pub(crate) const SUBBAND_START: [usize; 23] = [
    13, 19, 25, 31, 37, 49, 61, 73, 85, 97, 109, 121, 133, 145, 157, 169, 181, 193, 205, 217, 229,
    241, 253,
];

pub(crate) const MAX_BANDS: usize = 22;

/// Table E2.13. Index is the sub-band.
const DEFAULT_STRUCT: [bool; MAX_BANDS] = [
    false, false, false, false, false, false, false, false, false, true, false, true, false, true,
    false, true, true, true, false, true, true, true,
];

const AMP_EXP: [u8; 31] = [
    0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7,
];
const AMP_MANT: [u16; 31] = [
    0x20, 0x1b, 0x17, 0x13, 0x10, 0x1b, 0x17, 0x13, 0x10, 0x1b, 0x17, 0x13, 0x10, 0x1b, 0x17, 0x13,
    0x10, 0x1b, 0x17, 0x13, 0x10, 0x1b, 0x17, 0x13, 0x10, 0x1b, 0x17, 0x13, 0x10, 0x1b, 0x17,
];

/// Table E3.12, which is `-code / 7`.
const CHAOS: [f32; 8] = [
    0.0, -0.142_857, -0.285_714, -0.428_571, -0.571_429, -0.714_286, -0.857_143, -1.0,
];

/// One block's enhanced-coupling parameters, in coded units.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Setup {
    pub start: usize,
    pub end: usize,
    pub band_count: usize,
    pub band_size: [usize; MAX_BANDS],
    pub interp: bool,
    pub first: usize,
    pub amp: [[u8; MAX_BANDS]; MAX_FBW],
    pub angle: [[u8; MAX_BANDS]; MAX_FBW],
    pub chaos: [[u8; MAX_BANDS]; MAX_FBW],
    pub transient: [bool; MAX_FBW],
    pub coupled: [bool; MAX_FBW],
}

/// `ecpl_begin_subbnd` from `ecplbegf` (section 2.3.3.16).
pub(crate) fn begin_subband(code: usize) -> usize {
    if code < 3 {
        code * 2
    } else if code < 13 {
        code + 2
    } else {
        code * 2 - 10
    }
}

/// Sub-band one past the last coupled sub-band when spectral extension is off.
pub(crate) fn end_subband(code: usize) -> usize {
    code + 7
}

/// Table E2.13, used when the first enhanced-coupling block omits the structure.
pub(crate) fn default_structure() -> [bool; MAX_BANDS] {
    DEFAULT_STRUCT
}

/// Band count and the width of each band for sub-bands `begin..end`.
pub(crate) fn bands(
    begin: usize,
    end: usize,
    structure: &[bool; MAX_BANDS],
) -> (usize, [usize; MAX_BANDS]) {
    let mut count = 0usize;
    let mut size = [0usize; MAX_BANDS];
    for sub in begin..end.min(SUBBAND_START.len() - 1) {
        let combined = sub > begin && structure.get(sub).copied().unwrap_or(false);
        if !combined {
            count += 1;
        }
        if let Some(slot) = size.get_mut(count.saturating_sub(1)) {
            let width = SUBBAND_START[sub + 1] - SUBBAND_START[sub];
            *slot += width;
        }
    }
    (count, size)
}

/// Table E3.10. Code 31 is silence.
pub(crate) fn amplitude(code: u8) -> f32 {
    if code >= 31 {
        return 0.0;
    }
    let index = usize::from(code);
    let mantissa = f32::from(AMP_MANT[index]) / 32.0;
    mantissa / (1u32 << AMP_EXP[index]) as f32
}

/// Table E3.11, an angle in units of π, wrapped to `[-1, 1)`.
fn angle_code(code: u8) -> f32 {
    let code = u32::from(code) & 63;
    if code < 32 {
        code as f32 / 32.0
    } else {
        (code as f32 - 64.0) / 32.0
    }
}

fn wrap(mut value: f32) -> f32 {
    while value >= 1.0 {
        value -= 2.0;
    }
    while value < -1.0 {
        value += 2.0;
    }
    value
}

/// Uniform value in `[-1, 1)`, stable for the same inputs.
fn random(kind: u32, channel: usize, bin: usize, block: usize) -> f32 {
    let mut state = 0xA3C3_5EC0u32
        ^ kind
        ^ ((channel as u32) << 16)
        ^ ((bin as u32) << 8)
        ^ (block as u32).wrapping_mul(0x9E37_79B9);
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
}

fn fold(bin: usize) -> f32 {
    let n = SPAN as f32;
    let angle = 2.0 * PI * (n / 4.0 + 0.5) / n * (bin as f32 + 0.5);
    angle.cos()
}

/// Write each coupled channel's coefficients in `start..end`.
pub(crate) fn distribute(
    transform: &Transform,
    prev: &[f32; COEFS],
    curr: &[f32; COEFS],
    next: &[f32; COEFS],
    channels: &mut [[f32; COEFS]],
    setup: &Setup,
    block: usize,
) {
    if setup.start >= setup.end || setup.band_count == 0 {
        return;
    }
    let (zr, zi) = complex_channel(transform, prev, curr, next);
    for (ch, row) in channels.iter_mut().enumerate().take(MAX_FBW) {
        if !setup.coupled.get(ch).copied().unwrap_or(false) {
            continue;
        }
        let first = ch == setup.first;
        let (amp, angle) = bin_parameters(setup, ch, first);
        for bin in setup.start..setup.end.min(COEFS) {
            let chaos = if first {
                0.0
            } else {
                CHAOS[usize::from(setup.chaos_at(ch, bin)) & 7]
            };
            let draw = if setup.transient.get(ch).copied().unwrap_or(false) {
                random(1, ch, setup.band_at(bin), block)
            } else {
                random(0, ch, bin, 0)
            };
            let turned = wrap(angle[bin] + chaos * draw);
            let gain = amp[bin];
            let (sin, cos) = (turned * PI).sin_cos();
            let real = zr[bin] * gain * cos - zi[bin] * gain * sin;
            let imag = zi[bin] * gain * cos + zr[bin] * gain * sin;
            row[bin] = -2.0 * (fold(bin) * real + fold(SPAN / 2 - 1 - bin) * imag);
        }
    }
}

impl Setup {
    fn chaos_at(&self, channel: usize, bin: usize) -> u8 {
        self.chaos[channel][self.band_at(bin)]
    }

    fn band_at(&self, bin: usize) -> usize {
        let mut at = self.start;
        for (band, &size) in self.band_size.iter().enumerate().take(self.band_count) {
            if bin < at + size {
                return band;
            }
            at += size;
        }
        self.band_count.saturating_sub(1)
    }
}

fn bin_parameters(setup: &Setup, channel: usize, first: bool) -> ([f32; COEFS], [f32; COEFS]) {
    let mut amp = [0.0f32; COEFS];
    let mut angle = [0.0f32; COEFS];
    let transient = setup.transient[channel];
    let mut at = setup.start;
    for band in 0..setup.band_count {
        let mut gain = amplitude(setup.amp[channel][band]);
        if !transient && !first {
            let chaos = CHAOS[usize::from(setup.chaos[channel][band]) & 7];
            gain *= 1.0 + 0.38 * chaos;
        }
        let turned = if first {
            0.0
        } else {
            angle_code(setup.angle[channel][band])
        };
        let size = setup.band_size[band];
        for bin in at..at + size {
            if let Some(slot) = amp.get_mut(bin) {
                *slot = gain;
            }
            if let Some(slot) = angle.get_mut(bin) {
                *slot = turned;
            }
        }
        at += size;
    }
    if setup.interp && !first {
        interpolate(setup, channel, &mut angle);
    }
    (amp, angle)
}

/// Section 3.5.5.3. Bin 0 of the temporary buffer is the first coupled bin.
fn interpolate(setup: &Setup, channel: usize, angle: &mut [f32; COEFS]) {
    let mut relative = [0.0f32; COEFS];
    let sizes = &setup.band_size[..setup.band_count];
    if sizes.is_empty() {
        return;
    }
    let mut bin = 0isize;
    let mut slope = 0.0f32;
    let mut y = 0.0f32;
    for (band, &size) in sizes.iter().enumerate().skip(1) {
        let prev = sizes[band - 1];
        let previous = angle_code(setup.angle[channel][band - 1]);
        let mut current = angle_code(setup.angle[channel][band]);
        while current - previous > 1.0 {
            current -= 2.0;
        }
        while previous - current > 1.0 {
            current += 2.0;
        }
        let span = (size + prev) as f32 / 2.0;
        slope = if span == 0.0 {
            0.0
        } else {
            (current - previous) / span
        };
        if band == 1 && prev > 1 {
            let (start, origin) = if prev.is_multiple_of(2) {
                (previous - slope / 2.0, prev as isize / 2 - 1)
            } else {
                (previous - slope, (prev as isize - 3) / 2)
            };
            y = start;
            bin = origin;
            let count = bin + 1;
            for _ in 0..count {
                store(&mut relative, bin, y);
                y -= slope;
                bin -= 1;
            }
            bin = count;
        }
        let (origin, count) = if prev.is_multiple_of(2) {
            (previous + slope / 2.0, size / 2 + prev / 2)
        } else {
            (previous, size / 2 + prev.div_ceil(2))
        };
        y = origin;
        for _ in 0..count {
            store(&mut relative, bin, y);
            y += slope;
            bin += 1;
        }
    }
    let last = sizes[sizes.len() - 1];
    let count = if last.is_multiple_of(2) {
        last / 2
    } else {
        last / 2 + 1
    };
    for _ in 0..count {
        store(&mut relative, bin, y);
        y += slope;
        bin += 1;
    }
    for (bin, value) in relative.iter().enumerate().take(setup.end - setup.start) {
        if let Some(slot) = angle.get_mut(setup.start + bin) {
            *slot = *value;
        }
    }
}

fn store(angle: &mut [f32], index: isize, value: f32) {
    if index < 0 {
        return;
    }
    if let Some(slot) = angle.get_mut(index as usize) {
        *slot = wrap(value);
    }
}

/// Annex E 3.5.5.1 followed by the fold in 3.5.5.4.
fn complex_channel(
    transform: &Transform,
    prev: &[f32; COEFS],
    curr: &[f32; COEFS],
    next: &[f32; COEFS],
) -> ([f32; COEFS], [f32; COEFS]) {
    let mut previous = [0.0f32; SPAN];
    let mut current = [0.0f32; SPAN];
    let mut following = [0.0f32; SPAN];
    transform.inverse_long(prev, &mut previous);
    transform.inverse_long(curr, &mut current);
    transform.inverse_long(next, &mut following);
    let mut pcm = [0.0f32; SPAN];
    for n in 0..COEFS {
        pcm[n] = previous[n + COEFS] + current[n];
        pcm[n + COEFS] = current[n + COEFS] + following[n];
    }
    let window = transform.window_half();
    let mut real = [0.0f32; SPAN];
    let mut imag = [0.0f32; SPAN];
    for n in 0..SPAN {
        let weight = if n < COEFS {
            window[n]
        } else {
            window[SPAN - 1 - n]
        };
        let angle = PI * n as f32 / SPAN as f32;
        real[n] = pcm[n] * weight * angle.cos();
        imag[n] = pcm[n] * weight * -angle.sin();
    }
    let mut zr = [0.0f32; SPAN];
    let mut zi = [0.0f32; SPAN];
    transform.forward_dft(&real, &imag, &mut zr, &mut zi);
    let mut out_re = [0.0f32; COEFS];
    let mut out_im = [0.0f32; COEFS];
    out_re.copy_from_slice(&zr[..COEFS]);
    out_im.copy_from_slice(&zi[..COEFS]);
    (out_re, out_im)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn begin_codes_follow_the_three_ranges() {
        assert_eq!(begin_subband(0), 0);
        assert_eq!(begin_subband(2), 4);
        assert_eq!(begin_subband(3), 5);
        assert_eq!(begin_subband(12), 14);
        assert_eq!(begin_subband(13), 16);
        assert_eq!(end_subband(0), 7);
    }

    #[test]
    fn the_default_structure_keeps_the_low_subbands_separate() {
        let (count, size) = bands(0, 7, &default_structure());
        assert_eq!(count, 7);
        assert_eq!(&size[..4], &[6, 6, 6, 6]);
        assert_eq!(&size[4..7], &[12, 12, 12]);
        let (wide, _) = bands(8, 22, &default_structure());
        // Nine of those fourteen sub-bands join the band before them.
        assert_eq!(wide, 5);
    }

    #[test]
    fn amplitude_zero_is_unity_and_thirty_one_is_silent() {
        assert!((amplitude(0) - 1.0).abs() < 1e-6);
        assert!((amplitude(4) - 0.5).abs() < 1e-6);
        assert_eq!(amplitude(31), 0.0);
    }

    #[test]
    fn a_steady_coefficient_keeps_its_value_at_zero_angle() {
        let transform = Transform::new();
        let mut signal = [0.0f32; COEFS * 4];
        for (n, sample) in signal.iter_mut().enumerate() {
            *sample = (n as f32 * 0.17).sin() * 0.2;
        }
        let mut blocks = [[0.0f32; COEFS]; 3];
        for (block, coefs) in blocks.iter_mut().enumerate() {
            let mut input = [0.0f32; SPAN];
            input.copy_from_slice(&signal[block * COEFS..block * COEFS + SPAN]);
            transform.forward_long(&input, coefs);
        }
        let (count, size) = bands(4, 8, &DEFAULT_STRUCT);
        let mut setup = Setup {
            start: SUBBAND_START[4],
            end: SUBBAND_START[8],
            band_count: count,
            band_size: size,
            interp: false,
            first: 0,
            amp: [[0; MAX_BANDS]; MAX_FBW],
            angle: [[0; MAX_BANDS]; MAX_FBW],
            chaos: [[0; MAX_BANDS]; MAX_FBW],
            transient: [false; MAX_FBW],
            coupled: [false; MAX_FBW],
        };
        setup.coupled[0] = true;
        let mut channels = [[0.0f32; COEFS]; 1];
        channels[0][setup.start..setup.end].copy_from_slice(&blocks[1][setup.start..setup.end]);
        distribute(
            &transform,
            &blocks[0],
            &blocks[1],
            &blocks[2],
            &mut channels,
            &setup,
            1,
        );
        let mut worst = 0.0f32;
        for bin in setup.start..setup.end {
            worst = worst.max((channels[0][bin] - blocks[1][bin]).abs());
        }
        assert!(worst < 1.0e-4, "identity error {worst} at the coupled bins");
    }

    fn two_even_bands() -> Setup {
        let mut band_size = [0usize; MAX_BANDS];
        band_size[0] = 12;
        band_size[1] = 12;
        Setup {
            start: SUBBAND_START[4],
            end: SUBBAND_START[6],
            band_count: 2,
            band_size,
            interp: true,
            first: 0,
            amp: [[0; MAX_BANDS]; MAX_FBW],
            angle: [[0; MAX_BANDS]; MAX_FBW],
            chaos: [[0; MAX_BANDS]; MAX_FBW],
            transient: [false; MAX_FBW],
            coupled: [false; MAX_FBW],
        }
    }

    #[test]
    fn interpolation_holds_a_constant_angle_and_crosses_each_band_center() {
        let mut setup = two_even_bands();
        setup.angle[1][0] = 16;
        setup.angle[1][1] = 16;
        let (_, angle) = bin_parameters(&setup, 1, false);
        for (bin, value) in angle.iter().enumerate().take(setup.end).skip(setup.start) {
            assert!((value - 0.5).abs() < 1.0e-6, "bin {bin} drifted to {value}");
        }

        setup.angle[1][0] = 0;
        setup.angle[1][1] = 16;
        let slope = 0.5 / 12.0;
        let (_, angle) = bin_parameters(&setup, 1, false);
        let at = setup.start;
        assert!((angle[at + 5] - (0.0 - slope / 2.0)).abs() < 1.0e-5);
        assert!((angle[at + 6] - (slope / 2.0)).abs() < 1.0e-5);
        let center = (angle[at + 17] + angle[at + 18]) / 2.0;
        assert!((center - 0.5).abs() < 1.0e-5, "second band center {center}");
    }

    #[test]
    fn interpolation_takes_the_short_arc_across_the_branch_cut() {
        let mut setup = two_even_bands();
        // 29/32 and (35-64)/32. The short arc climbs through +1, not down through 0.
        setup.angle[1][0] = 29;
        setup.angle[1][1] = 35;
        let (_, angle) = bin_parameters(&setup, 1, false);
        let at = setup.start;
        assert!(angle[at + 6] > angle[at + 5]);
        assert!(angle[at + 5] > 0.8);
    }

    #[test]
    fn chaos_scales_only_a_steady_channel_that_is_not_first() {
        let mut setup = two_even_bands();
        setup.interp = false;
        setup.chaos[1][0] = 7;
        setup.chaos[1][1] = 7;
        let (amp, _) = bin_parameters(&setup, 1, false);
        assert!((amp[setup.start] - 0.62).abs() < 1.0e-5);

        setup.transient[1] = true;
        let (amp, _) = bin_parameters(&setup, 1, false);
        assert!((amp[setup.start] - 1.0).abs() < 1.0e-6);

        setup.transient[1] = false;
        setup.chaos[0][0] = 7;
        let (amp, angle) = bin_parameters(&setup, 0, true);
        assert!((amp[setup.start] - 1.0).abs() < 1.0e-6);
        assert_eq!(angle[setup.start], 0.0);
    }

    #[test]
    fn a_transient_draws_a_new_angle_each_block_and_a_steady_channel_does_not() {
        let transform = Transform::new();
        let mut signal = [0.0f32; COEFS * 4];
        for (n, sample) in signal.iter_mut().enumerate() {
            *sample = (n as f32 * 0.60).sin() * 0.5;
        }
        let mut blocks = [[0.0f32; COEFS]; 3];
        for (block, coefs) in blocks.iter_mut().enumerate() {
            let mut input = [0.0f32; SPAN];
            input.copy_from_slice(&signal[block * COEFS..block * COEFS + SPAN]);
            transform.forward_long(&input, coefs);
        }
        let mut setup = two_even_bands();
        setup.interp = false;
        setup.coupled[1] = true;
        setup.chaos[1][0] = 7;
        setup.chaos[1][1] = 7;
        let render = |setup: &Setup, block: usize| {
            let mut channels = [[0.0f32; COEFS]; 2];
            distribute(
                &transform,
                &blocks[0],
                &blocks[1],
                &blocks[2],
                &mut channels,
                setup,
                block,
            );
            channels[1]
        };
        let steady = render(&setup, 0);
        let steady_again = render(&setup, 4);
        let drift = steady
            .iter()
            .zip(&steady_again)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(drift < 1.0e-6, "steady chaos moved by {drift}");

        setup.transient[1] = true;
        let first = render(&setup, 0);
        let second = render(&setup, 1);
        let moved = first
            .iter()
            .zip(&second)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let energy: f32 = blocks[1][setup.start..setup.end]
            .iter()
            .map(|sample| sample * sample)
            .sum();
        assert!(
            moved > 1.0e-3,
            "transient chaos stayed put, delta {moved}, coupling energy {energy}"
        );
    }
}
