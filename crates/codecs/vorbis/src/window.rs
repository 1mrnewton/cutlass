//! The Vorbis window (section 4.3.1): flat in the middle, with power-
//! complementary slopes `sin(pi/2 sin^2(...))` where blocks overlap. A long
//! block next to a short one uses the short slope on that side.

use std::f64::consts::FRAC_PI_2;

/// The rising slopes for the short and long overlap.
#[derive(Clone, Debug)]
pub(crate) struct Windows {
    blocksizes: [usize; 2],
    /// `slopes[b][i]` for `i < blocksizes[b] / 2`.
    slopes: [Vec<f32>; 2],
}

/// Where the window of one block rises and falls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shape {
    pub n: usize,
    pub left_start: usize,
    pub left_end: usize,
    pub right_start: usize,
    pub right_end: usize,
    left: usize,
    right: usize,
}

impl Windows {
    pub(crate) fn new(blocksizes: [usize; 2]) -> Self {
        let slope = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|i| {
                    let inner = ((i as f64 + 0.5) / len as f64 * FRAC_PI_2).sin();
                    (FRAC_PI_2 * inner * inner).sin() as f32
                })
                .collect()
        };
        Self {
            blocksizes,
            slopes: [slope(blocksizes[0] / 2), slope(blocksizes[1] / 2)],
        }
    }

    /// The shape of a block from its own size and its neighbors' (section
    /// 4.3.1: a short block has short slopes on both sides).
    pub(crate) fn shape(&self, long: bool, previous_long: bool, next_long: bool) -> Shape {
        let n = self.blocksizes[usize::from(long)];
        let (left, right) = if long {
            (usize::from(previous_long), usize::from(next_long))
        } else {
            (0, 0)
        };
        let left_n = self.blocksizes[left] / 2;
        let right_n = self.blocksizes[right] / 2;
        Shape {
            n,
            left_start: n / 4 - left_n / 2,
            left_end: n / 4 + left_n / 2,
            right_start: 3 * n / 4 - right_n / 2,
            right_end: 3 * n / 4 + right_n / 2,
            left,
            right,
        }
    }

    /// Multiplies `block` (`shape.n` samples) by the window.
    pub(crate) fn apply(&self, shape: &Shape, block: &mut [f32]) {
        let rising = &self.slopes[shape.left];
        let falling = &self.slopes[shape.right];
        block[..shape.left_start].fill(0.0);
        for (value, gain) in block[shape.left_start..shape.left_end]
            .iter_mut()
            .zip(rising)
        {
            *value *= gain;
        }
        for (value, gain) in block[shape.right_start..shape.right_end]
            .iter_mut()
            .zip(falling.iter().rev())
        {
            *value *= gain;
        }
        block[shape.right_end..shape.n].fill(0.0);
    }

    /// The window value at `i`.
    #[cfg(test)]
    pub(crate) fn value(&self, shape: &Shape, i: usize) -> f32 {
        if i < shape.left_start || i >= shape.right_end {
            0.0
        } else if i < shape.left_end {
            self.slopes[shape.left][i - shape.left_start]
        } else if i < shape.right_start {
            1.0
        } else {
            self.slopes[shape.right][shape.right_end - 1 - i]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Windows;

    #[test]
    fn overlapping_slopes_are_power_complementary() {
        let windows = Windows::new([256, 2048]);
        for (long, prev, next) in [
            (true, true, true),
            (true, false, true),
            (false, false, false),
        ] {
            let shape = windows.shape(long, prev, next);
            let left = shape.left_end - shape.left_start;
            for i in 0..left {
                let rise = windows.value(&shape, shape.left_start + i);
                let fall = windows.value(&shape, shape.left_end - 1 - i);
                assert!((rise * rise + fall * fall - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn a_long_block_beside_short_ones_uses_short_slopes() {
        let windows = Windows::new([256, 2048]);
        let shape = windows.shape(true, false, true);
        assert_eq!((shape.left_start, shape.left_end), (448, 576));
        assert_eq!((shape.right_start, shape.right_end), (1024, 2048));
        let short = windows.shape(false, true, true);
        assert_eq!((short.left_start, short.left_end), (0, 128));
        assert_eq!((short.right_start, short.right_end), (128, 256));
        let mut block = vec![1.0; 2048];
        windows.apply(&shape, &mut block);
        assert_eq!(block[447], 0.0);
        assert_eq!(block[700], 1.0);
        for (i, &value) in block.iter().enumerate() {
            assert_eq!(value, windows.value(&shape, i), "sample {i}");
        }
    }
}
