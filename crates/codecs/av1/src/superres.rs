//! Horizontal upscaling used when a frame is coded smaller than it is shown.
//!
//! The phase and the eight taps are the specification's upscaling filter.
//! A constant input stays constant because every phase sums to 128.

const SCALE_BITS: i64 = 14;
const EXTRA_BITS: i64 = 8;
const FILTER_BITS: i32 = 7;

/// Eight taps for each of the 64 phases.
const FILTER: [[i16; 8]; 64] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, -1, 128, 2, -1, 0, 0],
    [0, 1, -3, 127, 4, -2, 1, 0],
    [0, 1, -4, 127, 6, -3, 1, 0],
    [0, 2, -6, 126, 8, -3, 1, 0],
    [0, 2, -7, 125, 11, -4, 1, 0],
    [-1, 2, -8, 125, 13, -5, 2, 0],
    [-1, 3, -9, 124, 15, -6, 2, 0],
    [-1, 3, -10, 123, 18, -6, 2, -1],
    [-1, 3, -11, 122, 20, -7, 3, -1],
    [-1, 4, -12, 121, 22, -8, 3, -1],
    [-1, 4, -13, 120, 25, -9, 3, -1],
    [-1, 4, -14, 118, 28, -9, 3, -1],
    [-1, 4, -15, 117, 30, -10, 4, -1],
    [-1, 5, -16, 116, 32, -11, 4, -1],
    [-1, 5, -16, 114, 35, -12, 4, -1],
    [-1, 5, -17, 112, 38, -12, 4, -1],
    [-1, 5, -18, 111, 40, -13, 5, -1],
    [-1, 5, -18, 109, 43, -14, 5, -1],
    [-1, 6, -19, 107, 45, -14, 5, -1],
    [-1, 6, -19, 105, 48, -15, 5, -1],
    [-1, 6, -19, 103, 51, -16, 5, -1],
    [-1, 6, -20, 101, 53, -16, 6, -1],
    [-1, 6, -20, 99, 56, -17, 6, -1],
    [-1, 6, -20, 97, 58, -17, 6, -1],
    [-1, 6, -20, 95, 61, -18, 6, -1],
    [-2, 7, -20, 93, 64, -18, 6, -2],
    [-2, 7, -20, 91, 66, -19, 6, -1],
    [-2, 7, -20, 88, 69, -19, 6, -1],
    [-2, 7, -20, 86, 71, -19, 6, -1],
    [-2, 7, -20, 84, 74, -20, 7, -2],
    [-2, 7, -20, 81, 76, -20, 7, -1],
    [-2, 7, -20, 79, 79, -20, 7, -2],
    [-1, 7, -20, 76, 81, -20, 7, -2],
    [-2, 7, -20, 74, 84, -20, 7, -2],
    [-1, 6, -19, 71, 86, -20, 7, -2],
    [-1, 6, -19, 69, 88, -20, 7, -2],
    [-1, 6, -19, 66, 91, -20, 7, -2],
    [-2, 6, -18, 64, 93, -20, 7, -2],
    [-1, 6, -18, 61, 95, -20, 6, -1],
    [-1, 6, -17, 58, 97, -20, 6, -1],
    [-1, 6, -17, 56, 99, -20, 6, -1],
    [-1, 6, -16, 53, 101, -20, 6, -1],
    [-1, 5, -16, 51, 103, -19, 6, -1],
    [-1, 5, -15, 48, 105, -19, 6, -1],
    [-1, 5, -14, 45, 107, -19, 6, -1],
    [-1, 5, -14, 43, 109, -18, 5, -1],
    [-1, 5, -13, 40, 111, -18, 5, -1],
    [-1, 4, -12, 38, 112, -17, 5, -1],
    [-1, 4, -12, 35, 114, -16, 5, -1],
    [-1, 4, -11, 32, 116, -16, 5, -1],
    [-1, 4, -10, 30, 117, -15, 4, -1],
    [-1, 3, -9, 28, 118, -14, 4, -1],
    [-1, 3, -9, 25, 120, -13, 4, -1],
    [-1, 3, -8, 22, 121, -12, 4, -1],
    [-1, 3, -7, 20, 122, -11, 3, -1],
    [-1, 2, -6, 18, 123, -10, 3, -1],
    [0, 2, -6, 15, 124, -9, 3, -1],
    [0, 2, -5, 13, 125, -8, 2, -1],
    [0, 1, -4, 11, 125, -7, 2, 0],
    [0, 1, -3, 8, 126, -6, 2, 0],
    [0, 1, -3, 6, 127, -4, 1, 0],
    [0, 1, -2, 4, 127, -3, 1, 0],
    [0, 0, -1, 2, 128, -1, 0, 0],
];

/// Coded width for an upscaled width and a super-resolution denominator.
///
/// The denominator is 8 when super resolution is off, and 9 through 16 when
/// it is on. The specification requires the coded width to be smaller.
#[must_use]
pub fn coded_width(upscaled: u32, denom: u32) -> u32 {
    upscaled.saturating_mul(8).saturating_add(denom / 2) / denom.max(1)
}

/// Upscale one plane from `down_w` samples to `up_w` samples.
///
/// `max_x` is the last sample the mi grid can address, including padding.
pub fn upscale(
    src: &[u8],
    stride: usize,
    down_w: usize,
    up_w: usize,
    height: usize,
    max_x: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; up_w.saturating_mul(height)];
    if up_w == 0 || height == 0 || stride == 0 || down_w == 0 {
        return out;
    }
    if up_w <= down_w {
        let n = down_w.min(stride);
        for y in 0..height {
            let from = y.saturating_mul(stride);
            let to = y.saturating_mul(up_w);
            let count = n.min(up_w);
            if from + count <= src.len() && to + count <= out.len() {
                out[to..to + count].copy_from_slice(&src[from..from + count]);
            }
        }
        return out;
    }
    let down = down_w as i64;
    let up = up_w as i64;
    let step = ((down << SCALE_BITS) + up / 2) / up;
    let err = up * step - (down << SCALE_BITS);
    let initial =
        ((-((up - down) << (SCALE_BITS - 1)) + up / 2) / up) + (1 << (EXTRA_BITS - 1)) - err / 2;
    let initial = initial & ((1 << SCALE_BITS) - 1);
    let last = max_x as i64;
    for y in 0..height {
        for x in 0..up_w {
            let src_x = -(1 << SCALE_BITS) + initial + (x as i64) * step;
            let base = src_x >> SCALE_BITS;
            let phase = ((src_x & ((1 << SCALE_BITS) - 1)) >> EXTRA_BITS) as usize;
            let mut sum = 0i32;
            for (k, tap) in FILTER[phase.min(63)].iter().enumerate() {
                let sample_x = (base + k as i64 - 3).clamp(0, last) as usize;
                let sample = i32::from(*src.get(y * stride + sample_x).unwrap_or(&0));
                sum += sample * i32::from(*tap);
            }
            let rounded = (sum + (1 << (FILTER_BITS - 1))) >> FILTER_BITS;
            out[y * up_w + x] = u8::try_from(rounded.clamp(0, 255)).unwrap_or(0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{FILTER, coded_width, upscale};

    #[test]
    fn every_phase_sums_to_unity() {
        assert_eq!(FILTER.len(), 64);
        for phase in FILTER {
            let sum: i32 = phase.iter().map(|tap| i32::from(*tap)).sum();
            assert_eq!(sum, 128);
        }
    }

    #[test]
    fn coded_width_shrinks_when_the_denominator_is_above_eight() {
        assert_eq!(coded_width(64, 8), 64);
        assert_eq!(coded_width(64, 16), 32);
        assert_eq!(coded_width(64, 9), (64 * 8 + 4) / 9);
    }

    #[test]
    fn a_constant_plane_stays_constant() {
        let src = vec![40u8; 16 * 4];
        let out = upscale(&src, 16, 8, 12, 4, 15);
        assert_eq!(out.len(), 12 * 4);
        assert!(out.iter().all(|sample| *sample == 40));
    }
}
