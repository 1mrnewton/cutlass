//! Intra sample prediction (clause 8.4.4.2): reference substitution,
//! filtering, planar, DC, and the 33 angular modes.

/// `intraPredAngle` for modes 0..=34 (Table 8-5). Modes 0 and 1 are unused.
const ANGLE: [i32; 35] = [
    0, 0, 32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32, -26, -21, -17, -13,
    -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32,
];

/// `invAngle` for modes 11..=25 (Table 8-6).
const INV_ANGLE: [i32; 15] = [
    -4096, -1638, -910, -630, -482, -390, -315, -256, -315, -390, -482, -630, -910, -1638, -4096,
];

/// Reference samples of one block, in the order of the substitution process:
/// `p[-1][2N-1]` up to `p[-1][-1]`, then `p[0][-1]` to `p[2N-1][-1]`.
pub struct Refs {
    pub samples: [i32; 129],
    pub available: [bool; 129],
    pub n: usize,
}

impl Refs {
    pub fn new(n: usize) -> Self {
        Self {
            samples: [0; 129],
            available: [false; 129],
            n,
        }
    }

    /// Index of `p[-1][y]`, `y` in `-1..2N`.
    #[inline]
    pub fn left(&self, y: i32) -> usize {
        (2 * self.n as i32 - 1 - y) as usize
    }

    /// Index of `p[x][-1]`, `x` in `-1..2N`.
    #[inline]
    pub fn top(&self, x: i32) -> usize {
        (2 * self.n as i32 + 1 + x) as usize
    }

    fn len(&self) -> usize {
        4 * self.n + 1
    }

    /// Clause 8.4.4.2.2.
    pub fn substitute(&mut self, bit_depth: u32) {
        let len = self.len();
        let Some(first) = self.available[..len].iter().position(|&a| a) else {
            self.samples[..len].fill(1 << (bit_depth - 1));
            return;
        };
        let value = self.samples[first];
        for i in 0..first {
            self.samples[i] = value;
        }
        for i in first + 1..len {
            if !self.available[i] {
                self.samples[i] = self.samples[i - 1];
            }
        }
    }
}

/// Clause 8.4.4.2.3, luma only for 4:2:0.
pub fn filter(refs: &mut Refs, mode: u8, strong_smoothing: bool, bit_depth: u32) {
    let n = refs.n;
    if mode == 1 || n == 4 {
        return;
    }
    let dist = (i32::from(mode) - 26)
        .abs()
        .min((i32::from(mode) - 10).abs());
    let threshold = match n {
        8 => 7,
        16 => 1,
        _ => 0,
    };
    if dist <= threshold {
        return;
    }
    let len = refs.len();
    let s = &refs.samples;
    let corner = s[refs.left(-1)];
    let bottom = s[refs.left(2 * n as i32 - 1)];
    let right = s[refs.top(2 * n as i32 - 1)];
    if strong_smoothing
        && n == 32
        && (corner + right - 2 * s[refs.top(n as i32 - 1)]).abs() < (1 << (bit_depth - 5))
        && (corner + bottom - 2 * s[refs.left(n as i32 - 1)]).abs() < (1 << (bit_depth - 5))
    {
        let mut out = [0i32; 129];
        out[refs.left(-1)] = corner;
        for y in 0..63 {
            out[refs.left(y)] = ((63 - y) * corner + (y + 1) * bottom + 32) >> 6;
        }
        out[refs.left(63)] = bottom;
        for x in 0..63 {
            out[refs.top(x)] = ((63 - x) * corner + (x + 1) * right + 32) >> 6;
        }
        out[refs.top(63)] = right;
        refs.samples[..len].copy_from_slice(&out[..len]);
        return;
    }
    let mut out = [0i32; 129];
    out[0] = s[0];
    out[len - 1] = s[len - 1];
    for i in 1..len - 1 {
        out[i] = (s[i - 1] + 2 * s[i] + s[i + 1] + 2) >> 2;
    }
    refs.samples[..len].copy_from_slice(&out[..len]);
}

/// Predict an `n`×`n` block into `dst` (row stride `stride`). `edge` turns
/// on the DC, horizontal, and vertical boundary smoothing of luma.
pub fn predict(refs: &Refs, mode: u8, edge: bool, bit_depth: u32, dst: &mut [u16], stride: usize) {
    let n = refs.n;
    let s = &refs.samples;
    let max = (1i32 << bit_depth) - 1;
    let p_left = |y: i32| s[refs.left(y)];
    let p_top = |x: i32| s[refs.top(x)];
    match mode {
        0 => {
            let shift = n.trailing_zeros() + 1;
            let ni = n as i32;
            let top_right = p_top(ni);
            let bottom_left = p_left(ni);
            for y in 0..ni {
                let row = &mut dst[y as usize * stride..y as usize * stride + n];
                let left = p_left(y);
                for x in 0..ni {
                    let value = (ni - 1 - x) * left
                        + (x + 1) * top_right
                        + (ni - 1 - y) * p_top(x)
                        + (y + 1) * bottom_left
                        + ni;
                    row[x as usize] = (value >> shift) as u16;
                }
            }
        }
        1 => {
            let ni = n as i32;
            let mut sum = ni;
            for i in 0..ni {
                sum += p_top(i) + p_left(i);
            }
            let dc = sum >> (n.trailing_zeros() + 1);
            for y in 0..n {
                dst[y * stride..y * stride + n].fill(dc as u16);
            }
            if edge && n < 32 {
                dst[0] = ((p_left(0) + 2 * dc + p_top(0) + 2) >> 2) as u16;
                for x in 1..ni {
                    dst[x as usize] = ((p_top(x) + 3 * dc + 2) >> 2) as u16;
                }
                for y in 1..ni {
                    dst[y as usize * stride] = ((p_left(y) + 3 * dc + 2) >> 2) as u16;
                }
            }
        }
        _ => angular(refs, mode, edge, max, dst, stride),
    }
}

fn angular(refs: &Refs, mode: u8, edge: bool, max: i32, dst: &mut [u16], stride: usize) {
    let n = refs.n as i32;
    let s = &refs.samples;
    let angle = ANGLE[usize::from(mode)];
    // ref[] runs from -n to 2n; store it at an offset of n.
    let mut line = [0i32; 3 * 64 + 1];
    let base = n as usize;
    let vertical = mode >= 18;
    let main = |i: i32| {
        if vertical {
            s[refs.top(i)]
        } else {
            s[refs.left(i)]
        }
    };
    let side = |i: i32| {
        if vertical {
            s[refs.left(i)]
        } else {
            s[refs.top(i)]
        }
    };
    for x in 0..=n {
        line[(base as i32 + x) as usize] = main(x - 1);
    }
    if angle < 0 {
        let last = (n * angle) >> 5;
        if last < -1 {
            let inv = INV_ANGLE[usize::from(mode) - 11];
            for x in last..=-1 {
                line[(base as i32 + x) as usize] = side(-1 + ((x * inv + 128) >> 8));
            }
        }
    } else {
        for x in n + 1..=2 * n {
            line[(base as i32 + x) as usize] = main(x - 1);
        }
    }
    for y in 0..n {
        let pos = (y + 1) * angle;
        let idx = pos >> 5;
        let fact = pos & 31;
        for x in 0..n {
            let at = (base as i32 + x + idx + 1) as usize;
            let value = if fact != 0 {
                ((32 - fact) * line[at] + fact * line[at + 1] + 16) >> 5
            } else {
                line[at]
            };
            // For a horizontal mode the roles of x and y swap.
            let (px, py) = if vertical { (x, y) } else { (y, x) };
            dst[py as usize * stride + px as usize] = value as u16;
        }
    }
    if edge && n < 32 {
        if mode == 26 {
            let corner = s[refs.left(-1)];
            let top = s[refs.top(0)];
            for y in 0..n {
                let value = top + ((s[refs.left(y)] - corner) >> 1);
                dst[y as usize * stride] = value.clamp(0, max) as u16;
            }
        } else if mode == 10 {
            let corner = s[refs.left(-1)];
            let left = s[refs.left(0)];
            for x in 0..n {
                let value = left + ((s[refs.top(x)] - corner) >> 1);
                dst[x as usize] = value.clamp(0, max) as u16;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Refs, predict};

    fn flat(n: usize, value: i32) -> Refs {
        let mut refs = Refs::new(n);
        for i in 0..4 * n + 1 {
            refs.samples[i] = value;
            refs.available[i] = true;
        }
        refs
    }

    #[test]
    fn every_mode_of_flat_references_is_flat() {
        for n in [4usize, 8, 16, 32] {
            let refs = flat(n, 77);
            for mode in 0..35u8 {
                let mut out = vec![0u16; n * n];
                predict(&refs, mode, true, 8, &mut out, n);
                assert!(out.iter().all(|&v| v == 77), "n {n} mode {mode}");
            }
        }
    }

    #[test]
    fn nothing_available_is_mid_grey() {
        let mut refs = Refs::new(8);
        refs.substitute(10);
        assert!(refs.samples[..33].iter().all(|&v| v == 512));
    }

    #[test]
    fn substitution_copies_the_nearest_earlier_sample() {
        let mut refs = Refs::new(4);
        refs.samples[5] = 9;
        refs.available[5] = true;
        refs.samples[7] = 3;
        refs.available[7] = true;
        refs.substitute(8);
        assert_eq!(&refs.samples[..10], &[9, 9, 9, 9, 9, 9, 9, 3, 3, 3]);
    }

    #[test]
    fn vertical_copies_the_row_above() {
        let mut refs = flat(4, 0);
        for x in 0..8 {
            let at = refs.top(x);
            refs.samples[at] = 10 * (x + 1);
        }
        let mut out = vec![0u16; 16];
        predict(&refs, 26, false, 8, &mut out, 4);
        assert_eq!(&out[12..16], &[10, 20, 30, 40]);
    }
}
