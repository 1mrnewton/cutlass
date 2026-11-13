//! Intra sample prediction (clause 8.4.4.2): reference substitution,
//! filtering, planar, DC, and the 33 angular modes.

use crate::pixel::Pixel;

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
pub fn predict<P: Pixel>(
    refs: &Refs,
    mode: u8,
    edge: bool,
    bit_depth: u32,
    dst: &mut [P],
    stride: usize,
) {
    let max = (1i32 << bit_depth) - 1;
    match refs.n {
        4 => predict_n::<P, 4>(refs, mode, edge, max, dst, stride),
        8 => predict_n::<P, 8>(refs, mode, edge, max, dst, stride),
        16 => predict_n::<P, 16>(refs, mode, edge, max, dst, stride),
        _ => predict_n::<P, 32>(refs, mode, edge, max, dst, stride),
    }
}

/// [`predict`] of an `N`×`N` block: each row of the prediction is
/// computed over whole slices of the references, which the compiler turns
/// into vector code.
#[inline(always)]
fn predict_n<P: Pixel, const N: usize>(
    refs: &Refs,
    mode: u8,
    edge: bool,
    max: i32,
    dst: &mut [P],
    stride: usize,
) {
    let s = &refs.samples;
    // `p[x][-1]` for x in 0..2N, and `p[-1][y]` at `s[2N - 1 - y]`.
    let top = &s[2 * N + 1..4 * N + 1];
    let left = |y: usize| s[2 * N - 1 - y];
    let corner = s[2 * N];
    let rows = dst.chunks_mut(stride).take(N);
    match mode {
        0 => {
            let shift = N.trailing_zeros() + 1;
            let n = N as i32;
            let (top_right, bottom_left) = (top[N], left(N));
            for (y, row) in rows.enumerate() {
                let l = left(y);
                let below = n - 1 - y as i32;
                let add = (y as i32 + 1) * bottom_left + n;
                for (x, (d, &t)) in row[..N].iter_mut().zip(&top[..N]).enumerate() {
                    let x = x as i32;
                    let v = (n - 1 - x) * l + (x + 1) * top_right + below * t + add;
                    *d = P::of(v >> shift);
                }
            }
        }
        1 => {
            let mut sum = N as i32;
            for (i, &t) in top[..N].iter().enumerate() {
                sum += t + left(i);
            }
            let dc = sum >> (N.trailing_zeros() + 1);
            for row in rows {
                row[..N].fill(P::of(dc));
            }
            if edge && N < 32 {
                dst[0] = P::of((left(0) + 2 * dc + top[0] + 2) >> 2);
                for x in 1..N {
                    dst[x] = P::of((top[x] + 3 * dc + 2) >> 2);
                }
                for y in 1..N {
                    dst[y * stride] = P::of((left(y) + 3 * dc + 2) >> 2);
                }
            }
        }
        _ => {
            angular::<P, N>(refs, mode, dst, stride);
            if edge && N < 32 && (mode == 26 || mode == 10) {
                let other = if mode == 26 { top[0] } else { left(0) };
                for i in 0..N {
                    let side = if mode == 26 { left(i) } else { top[i] };
                    let v = P::of((other + ((side - corner) >> 1)).clamp(0, max));
                    if mode == 26 {
                        dst[i * stride] = v;
                    } else {
                        dst[i] = v;
                    }
                }
            }
        }
    }
}

/// The angular modes (clause 8.4.4.2.6) of an `N`×`N` block. A horizontal
/// mode is the vertical one with x and y swapped: its rows are computed as
/// columns and then written across.
#[inline(always)]
fn angular<P: Pixel, const N: usize>(refs: &Refs, mode: u8, dst: &mut [P], stride: usize) {
    let s = &refs.samples;
    let angle = ANGLE[usize::from(mode)];
    let vertical = mode >= 18;
    // `main(i)`, i in -1..2N: the references along the direction, `side`
    // the others; `p[x][-1]` is at `s[2N + 1 + x]`, `p[-1][y]` at
    // `s[2N - 1 - y]`.
    let main = |i: i32| s[(2 * N as i32 + if vertical { 1 + i } else { -1 - i }) as usize];
    let side = |i: i32| s[(2 * N as i32 + if vertical { -1 - i } else { 1 + i }) as usize];
    // ref[] runs from -N to 2N; it is stored at an offset of N, with one
    // more sample that a row with `fact` 0 multiplies by 0.
    let mut line = [0i32; 3 * 32 + 2];
    let n = N as i32;
    for x in 0..=n {
        line[(n + x) as usize] = main(x - 1);
    }
    if angle < 0 {
        let last = (n * angle) >> 5;
        if last < -1 {
            let inv = INV_ANGLE[usize::from(mode) - 11];
            for x in last..=-1 {
                line[(n + x) as usize] = side(-1 + ((x * inv + 128) >> 8));
            }
        }
    } else {
        for x in n + 1..=2 * n {
            line[(n + x) as usize] = main(x - 1);
        }
    }
    // Row `y` of the direction: `fact` 0 gives `line[at]` by the formula.
    let row = |y: usize, out: &mut [i32; N]| {
        let pos = (y as i32 + 1) * angle;
        let (idx, fact) = (pos >> 5, pos & 31);
        let at = (n + idx + 1) as usize;
        let (a, b) = (&line[at..at + N], &line[at + 1..at + 1 + N]);
        for ((o, &a), &b) in out.iter_mut().zip(a).zip(b) {
            *o = ((32 - fact) * a + fact * b + 16) >> 5;
        }
    };
    if vertical {
        let mut out = [0i32; N];
        for (y, d) in dst.chunks_mut(stride).take(N).enumerate() {
            row(y, &mut out);
            for (d, &v) in d[..N].iter_mut().zip(&out) {
                *d = P::of(v);
            }
        }
    } else {
        let mut block = [[0i32; N]; N];
        for (y, out) in block.iter_mut().enumerate() {
            row(y, out);
        }
        for (x, d) in dst.chunks_mut(stride).take(N).enumerate() {
            for (d, column) in d[..N].iter_mut().zip(&block) {
                *d = P::of(column[x]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ANGLE, INV_ANGLE, Pixel, Refs, predict};

    /// The prediction as first written, sample by sample from the clause.
    pub fn predict_reference<P: Pixel>(
        refs: &Refs,
        mode: u8,
        edge: bool,
        bit_depth: u32,
        dst: &mut [P],
        stride: usize,
    ) {
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
                        row[x as usize] = P::of(value >> shift);
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
                    dst[y * stride..y * stride + n].fill(P::of(dc));
                }
                if edge && n < 32 {
                    dst[0] = P::of((p_left(0) + 2 * dc + p_top(0) + 2) >> 2);
                    for x in 1..ni {
                        dst[x as usize] = P::of((p_top(x) + 3 * dc + 2) >> 2);
                    }
                    for y in 1..ni {
                        dst[y as usize * stride] = P::of((p_left(y) + 3 * dc + 2) >> 2);
                    }
                }
            }
            _ => angular_reference(refs, mode, edge, max, dst, stride),
        }
    }

    fn angular_reference<P: Pixel>(
        refs: &Refs,
        mode: u8,
        edge: bool,
        max: i32,
        dst: &mut [P],
        stride: usize,
    ) {
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
                dst[py as usize * stride + px as usize] = P::of(value);
            }
        }
        if edge && n < 32 {
            if mode == 26 {
                let corner = s[refs.left(-1)];
                let top = s[refs.top(0)];
                for y in 0..n {
                    let value = top + ((s[refs.left(y)] - corner) >> 1);
                    dst[y as usize * stride] = P::of(value.clamp(0, max));
                }
            } else if mode == 10 {
                let corner = s[refs.left(-1)];
                let left = s[refs.left(0)];
                for x in 0..n {
                    let value = left + ((s[refs.top(x)] - corner) >> 1);
                    dst[x as usize] = P::of(value.clamp(0, max));
                }
            }
        }
    }

    #[test]
    fn matches_the_sample_by_sample_prediction() {
        let mut seed = 0x1a7_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for n in [4usize, 8, 16, 32] {
            for bit_depth in [8u32, 10, 12] {
                for round in 0..20 {
                    let mut refs = Refs::new(n);
                    let max = (1u64 << bit_depth) - 1;
                    for i in 0..4 * n + 1 {
                        refs.samples[i] = match round % 4 {
                            0 => max as i32,
                            1 => (next() % 2 * max) as i32,
                            _ => (next() % (max + 1)) as i32,
                        };
                    }
                    for mode in 0..35u8 {
                        for edge in [false, true] {
                            let stride = n + 3;
                            let mut want = vec![7u16; n * stride];
                            let mut got = want.clone();
                            predict_reference(&refs, mode, edge, bit_depth, &mut want, stride);
                            predict(&refs, mode, edge, bit_depth, &mut got, stride);
                            assert_eq!(
                                got, want,
                                "n {n} depth {bit_depth} mode {mode} edge {edge}"
                            );
                        }
                    }
                }
            }
        }
    }

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
