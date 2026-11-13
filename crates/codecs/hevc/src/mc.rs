//! Fractional sample interpolation (clause 8.5.3.3.3) and weighted sample
//! prediction (clause 8.5.3.3.4).

/// Luma 8-tap filter `fL` for quarter positions 1..=3.
const LUMA: [[i32; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
];

/// Chroma 4-tap filter `fC` for eighth positions 1..=7.
const CHROMA: [[i32; 4]; 8] = [
    [0, 64, 0, 0],
    [-2, 58, 10, -2],
    [-4, 54, 16, -2],
    [-6, 46, 28, -4],
    [-4, 36, 36, -4],
    [-4, 28, 46, -6],
    [-2, 16, 54, -4],
    [-2, 10, 58, -2],
];

/// A reference plane.
pub struct Plane<'a> {
    pub samples: &'a [u16],
    pub width: usize,
    pub height: usize,
}

/// Interpolate a `w`×`h` block whose integer origin is (`x`, `y`) at
/// fraction (`fx`, `fy`) into `out` (stride `w`), 14-bit intermediate
/// precision. Luma uses the 8-tap filter in quarters, chroma the 4-tap
/// filter in eighths. Samples outside the plane repeat its edge.
///
/// The row loops are written to compile to vector code; call this inside
/// [`cutlass_simd::wide::wide`] to get 256-bit vectors where the CPU has
/// them.
#[inline(always)]
pub fn interpolate(
    plane: &Plane<'_>,
    block: Block,
    bit_depth: u32,
    out: &mut [i16],
    scratch: &mut Scratch,
) {
    if block.luma {
        let taps = [LUMA16[block.fx & 3], LUMA16[block.fy & 3]];
        filter::<8>(plane, block, bit_depth, taps, (out, scratch));
    } else {
        let taps = [CHROMA16[block.fx & 7], CHROMA16[block.fy & 7]];
        filter::<4>(plane, block, bit_depth, taps, (out, scratch));
    }
}

const LUMA16: [[i16; 8]; 4] = {
    let mut t = [[0i16; 8]; 4];
    let mut f = 0;
    while f < 4 {
        let mut k = 0;
        while k < 8 {
            t[f][k] = LUMA[f][k] as i16;
            k += 1;
        }
        f += 1;
    }
    t
};

const CHROMA16: [[i16; 4]; 8] = {
    let mut t = [[0i16; 4]; 8];
    let mut f = 0;
    while f < 8 {
        let mut k = 0;
        while k < 4 {
            t[f][k] = CHROMA[f][k] as i16;
            k += 1;
        }
        f += 1;
    }
    t
};

/// Buffers the interpolation reuses from block to block.
pub struct Scratch {
    win: Vec<u16>,
    tmp: Vec<i16>,
}

impl Default for Scratch {
    fn default() -> Self {
        Self {
            win: vec![0; WINDOW * WINDOW],
            tmp: vec![0; WINDOW * 64],
        }
    }
}

/// Largest window a 64×64 block with an 8-tap filter reads, per edge.
const WINDOW: usize = 64 + 7;

#[inline(always)]
fn filter<const T: usize>(
    plane: &Plane<'_>,
    block: Block,
    bit_depth: u32,
    [cx, cy]: [[i16; T]; 2],
    (out, scratch): (&mut [i16], &mut Scratch),
) {
    let Block {
        x, y, fx, fy, w, h, ..
    } = block;
    let (w, h) = (w.min(64), h.min(64));
    let half = (T / 2 - 1) as isize;
    let shift1 = bit_depth.saturating_sub(8).min(4);
    let shift3 = 14u32.saturating_sub(bit_depth).max(2);
    let sw = w + T - 1;
    let sh = h + T - 1;
    let (pw, ph) = (plane.width as isize, plane.height as isize);
    let (x0, y0) = (x - half, y - half);
    let out = &mut out[..w * h];
    if pw == 0 || ph == 0 || plane.samples.len() < plane.width * plane.height {
        out.fill(0);
        return;
    }
    let inside = x0 >= 0 && y0 >= 0 && x0 + sw as isize <= pw && y0 + sh as isize <= ph;
    let Scratch { win, tmp } = scratch;
    // The source rows: the reference itself, or a window with its edges
    // repeated when the block reaches outside.
    let (src, stride, base): (&[u16], usize, usize) = if inside {
        (
            plane.samples,
            plane.width,
            y0 as usize * plane.width + x0 as usize,
        )
    } else {
        pad(plane, (x0, y0), sw, &mut win[..sw * sh]);
        (&*win, sw, 0)
    };
    let row = |j: usize| -> &[u16] { &src[base + j * stride..][..sw] };
    let h0 = T / 2 - 1;
    // Eight-bit samples times the taps fit 16 bits (at most 88 · 255).
    let narrow = bit_depth == 8;
    match (fx != 0, fy != 0) {
        (false, false) => {
            for (j, dst) in out.chunks_exact_mut(w).enumerate() {
                for (d, &s) in dst.iter_mut().zip(&row(j + h0)[h0..]) {
                    *d = (s << shift3) as i16;
                }
            }
        }
        (true, false) => {
            for (j, dst) in out.chunks_exact_mut(w).enumerate() {
                let line = row(j + h0);
                let lines: [&[u16]; T] = std::array::from_fn(|k| &line[k..k + w]);
                taps_u16(&cx, lines, dst, narrow, shift1);
            }
        }
        (false, true) => {
            for (j, dst) in out.chunks_exact_mut(w).enumerate() {
                let lines: [&[u16]; T] = std::array::from_fn(|k| &row(j + k)[h0..h0 + w]);
                taps_u16(&cy, lines, dst, narrow, shift1);
            }
        }
        (true, true) => {
            let tmp = &mut tmp[..sh * w];
            for (j, dst) in tmp.chunks_exact_mut(w).enumerate() {
                let line = row(j);
                let lines: [&[u16]; T] = std::array::from_fn(|k| &line[k..k + w]);
                taps_u16(&cx, lines, dst, narrow, shift1);
            }
            let tmp = &*tmp;
            for (j, dst) in out.chunks_exact_mut(w).enumerate() {
                let lines: [&[i16]; T] = std::array::from_fn(|k| &tmp[(j + k) * w..(j + k + 1) * w]);
                taps_i16(&cy, lines, dst);
            }
        }
    }
}

/// Copy the `sw`×`sh` window at (`x0`, `y0`) of `plane` into `win`, the
/// samples outside the plane repeating its nearest edge.
#[inline(always)]
fn pad(plane: &Plane<'_>, (x0, y0): (isize, isize), sw: usize, win: &mut [u16]) {
    let (pw, ph) = (plane.width as isize, plane.height as isize);
    // Columns of the window inside the plane: `a..b`.
    let a = (-x0).clamp(0, sw as isize) as usize;
    let b = (pw - x0).clamp(0, sw as isize) as usize;
    for (j, dst) in win.chunks_exact_mut(sw).enumerate() {
        let sy = (y0 + j as isize).clamp(0, ph - 1) as usize * plane.width;
        let line = &plane.samples[sy..sy + plane.width];
        if a < b {
            let from = (x0 + a as isize) as usize;
            dst[a..b].copy_from_slice(&line[from..from + (b - a)]);
            let left = line[0];
            let right = line[plane.width - 1];
            dst[..a].fill(left);
            dst[b..].fill(right);
        } else {
            // Wholly left or right of the plane: its nearest edge sample.
            let v = if x0 >= pw { line[plane.width - 1] } else { line[0] };
            dst.fill(v);
        }
    }
}

/// Run `$f::<L>(i, out)` over `$out` in runs of 16, then 8, 4 and 2
/// outputs (every block width is even), `i` the first index of a run.
macro_rules! by_lanes {
    ($out:expr, |$i:ident, $o:ident| $body16:expr, $body8:expr, $body4:expr, $body2:expr) => {{
        let out: &mut [i16] = $out;
        let (runs, rest) = out.as_chunks_mut::<16>();
        let mut $i = 0usize;
        for $o in runs {
            $body16;
            $i += 16;
        }
        let (runs, rest) = rest.as_chunks_mut::<8>();
        for $o in runs {
            $body8;
            $i += 8;
        }
        let (runs, rest) = rest.as_chunks_mut::<4>();
        for $o in runs {
            $body4;
            $i += 4;
        }
        let (runs, _) = rest.as_chunks_mut::<2>();
        for $o in runs {
            $body2;
            $i += 2;
        }
    }};
}

/// `out[i]` = the sum over `k` of `c[k] · lines[k][i]`, shifted right by
/// `shift`. With `narrow` the sums are taken in 16 bits, which holds them
/// for 8-bit samples (and then `shift` is 0).
#[inline(always)]
fn taps_u16<const T: usize>(
    c: &[i16; T],
    lines: [&[u16]; T],
    out: &mut [i16],
    narrow: bool,
    shift: u32,
) {
    let n = out.len();
    let lines: [&[u16]; T] = lines.map(|l| &l[..n]);
    if narrow {
        by_lanes!(out, |i, o| lanes_narrow::<16, T>(c, &lines, i, o),
            lanes_narrow::<8, T>(c, &lines, i, o),
            lanes_narrow::<4, T>(c, &lines, i, o),
            lanes_narrow::<2, T>(c, &lines, i, o));
    } else {
        by_lanes!(out, |i, o| lanes_wide::<16, T, u16>(c, &lines, i, shift, o),
            lanes_wide::<8, T, u16>(c, &lines, i, shift, o),
            lanes_wide::<4, T, u16>(c, &lines, i, shift, o),
            lanes_wide::<2, T, u16>(c, &lines, i, shift, o));
    }
}

/// The second, vertical pass over 14-bit intermediates: the sums shifted
/// right by 6.
#[inline(always)]
fn taps_i16<const T: usize>(c: &[i16; T], lines: [&[i16]; T], out: &mut [i16]) {
    let n = out.len();
    let lines: [&[i16]; T] = lines.map(|l| &l[..n]);
    by_lanes!(out, |i, o| lanes_wide::<16, T, i16>(c, &lines, i, 6, o),
        lanes_wide::<8, T, i16>(c, &lines, i, 6, o),
        lanes_wide::<4, T, i16>(c, &lines, i, 6, o),
        lanes_wide::<2, T, i16>(c, &lines, i, 6, o));
}

/// `L` samples of `line` from `i`; zeros past its end (never reached:
/// every line holds the outputs' width).
#[inline(always)]
fn run<const L: usize, S: Copy + Default>(line: &[S], i: usize) -> [S; L] {
    match line.get(i..).and_then(|l| l.first_chunk::<L>()) {
        Some(a) => *a,
        None => [S::default(); L],
    }
}

/// `L` outputs from index `i` with 16-bit sums of 8-bit samples.
#[inline(always)]
fn lanes_narrow<const L: usize, const T: usize>(
    c: &[i16; T],
    lines: &[&[u16]; T],
    i: usize,
    out: &mut [i16; L],
) {
    let mut acc = [0i16; L];
    for k in 0..T {
        let src = run::<L, u16>(lines[k], i);
        let ck = c[k];
        for l in 0..L {
            acc[l] = acc[l].wrapping_add(ck.wrapping_mul(src[l] as i16));
        }
    }
    *out = acc;
}

/// `L` outputs from index `i` with 32-bit sums, shifted right by `shift`.
#[inline(always)]
fn lanes_wide<const L: usize, const T: usize, S: Copy + Default>(
    c: &[i16; T],
    lines: &[&[S]; T],
    i: usize,
    shift: u32,
    out: &mut [i16; L],
) where
    i32: From<S>,
{
    let mut acc = [0i32; L];
    for k in 0..T {
        let src = run::<L, S>(lines[k], i);
        let ck = c[k] as i32;
        for l in 0..L {
            acc[l] += ck * i32::from(src[l]);
        }
    }
    for l in 0..L {
        out[l] = (acc[l] >> shift) as i16;
    }
}

/// Where and how large one prediction block is in a reference plane.
#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub x: isize,
    pub y: isize,
    pub fx: usize,
    pub fy: usize,
    pub w: usize,
    pub h: usize,
    pub luma: bool,
}

/// Explicit weight of one list: weight, offset (already scaled to the bit
/// depth), and `log2Wd` without the 14-bit shift.
#[derive(Clone, Copy, Debug)]
pub struct Wp {
    pub weight: i32,
    pub offset: i32,
    pub log2_denom: u32,
}

/// Write one predicted block: default or explicit weighting of one or two
/// interpolated blocks (clause 8.5.3.3.4).
#[inline(always)]
pub fn store(
    dst: &mut [u16],
    stride: usize,
    w: usize,
    h: usize,
    bit_depth: u32,
    pred: [Option<&[i16]>; 2],
    weights: Option<[Wp; 2]>,
) {
    let max = (1i32 << bit_depth) - 1;
    let shift1 = 14 - bit_depth;
    let rows = dst.chunks_mut(stride).take(h);
    match (pred, weights) {
        ([Some(a), Some(b)], None) => {
            let shift2 = 15 - bit_depth;
            let add = 1i32 << (shift2 - 1);
            for (row, (a, b)) in rows.zip(a.chunks_exact(w).zip(b.chunks_exact(w))) {
                for (out, (&a, &b)) in row[..w].iter_mut().zip(a.iter().zip(b)) {
                    let v = (i32::from(a) + i32::from(b) + add) >> shift2;
                    *out = v.clamp(0, max) as u16;
                }
            }
        }
        ([Some(a), Some(b)], Some([w0, w1])) => {
            let log2wd = w0.log2_denom + shift1;
            let add = (w0.offset + w1.offset + 1) << log2wd;
            for (row, (a, b)) in rows.zip(a.chunks_exact(w).zip(b.chunks_exact(w))) {
                for (out, (&a, &b)) in row[..w].iter_mut().zip(a.iter().zip(b)) {
                    let v =
                        (i32::from(a) * w0.weight + i32::from(b) * w1.weight + add) >> (log2wd + 1);
                    *out = v.clamp(0, max) as u16;
                }
            }
        }
        ([Some(a), None] | [None, Some(a)], None) => {
            let add = 1i32 << (shift1 - 1);
            for (row, a) in rows.zip(a.chunks_exact(w)) {
                for (out, &a) in row[..w].iter_mut().zip(a) {
                    let v = (i32::from(a) + add) >> shift1;
                    *out = v.clamp(0, max) as u16;
                }
            }
        }
        ([Some(a), None], Some([wp, _])) | ([None, Some(a)], Some([_, wp])) => {
            let log2wd = wp.log2_denom + shift1;
            for (row, a) in rows.zip(a.chunks_exact(w)) {
                for (out, &a) in row[..w].iter_mut().zip(a) {
                    let p = i32::from(a);
                    let v = if log2wd >= 1 {
                        ((p * wp.weight + (1 << (log2wd - 1))) >> log2wd) + wp.offset
                    } else {
                        p * wp.weight + wp.offset
                    };
                    *out = v.clamp(0, max) as u16;
                }
            }
        }
        ([None, None], _) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::{Block, Plane, Scratch, Wp, interpolate, store};

    #[test]
    fn whole_sample_motion_copies_the_reference() {
        let samples: Vec<u16> = (0..64).map(|v| v as u16).collect();
        let plane = Plane {
            samples: &samples,
            width: 8,
            height: 8,
        };
        let mut out = vec![0i16; 4];
        let block = Block {
            x: 2,
            y: 3,
            fx: 0,
            fy: 0,
            w: 2,
            h: 2,
            luma: true,
        };
        interpolate(&plane, block, 8, &mut out, &mut Scratch::default());
        let mut dst = vec![0u16; 4];
        store(&mut dst, 2, 2, 2, 8, [Some(&out), None], None);
        assert_eq!(dst, vec![26, 27, 34, 35]);
    }

    #[test]
    fn half_sample_of_a_flat_plane_is_flat_and_edges_clamp() {
        let samples = vec![100u16; 16];
        let plane = Plane {
            samples: &samples,
            width: 4,
            height: 4,
        };
        let mut out = vec![0i16; 16];
        let block = Block {
            x: -9,
            y: 7,
            fx: 2,
            fy: 3,
            w: 4,
            h: 4,
            luma: true,
        };
        interpolate(&plane, block, 10, &mut out, &mut Scratch::default());
        let mut dst = vec![0u16; 16];
        store(&mut dst, 4, 4, 4, 10, [Some(&out), Some(&out)], None);
        assert!(dst.iter().all(|&v| v == 100));
        let wp = Wp {
            weight: 2,
            offset: 4,
            log2_denom: 1,
        };
        store(&mut dst, 4, 4, 4, 10, [Some(&out), None], Some([wp, wp]));
        assert!(dst.iter().all(|&v| v == 104));
    }
}
