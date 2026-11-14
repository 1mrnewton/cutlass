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

pub use cutlass_simd::hevc::mc::Weights;
use cutlass_simd::hevc::mc::{self as kernels, Dst, Src, Taps};

use crate::pixel::Pixel;

/// A reference plane: `rows` of its `height` rows are in `samples`.
pub struct Plane<'a, P> {
    pub samples: &'a [P],
    pub width: usize,
    pub height: usize,
    pub rows: usize,
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
pub fn interpolate<P: Pixel>(
    plane: &Plane<'_, P>,
    block: Block,
    bit_depth: u32,
    out: &mut [i16],
    scratch: &mut Scratch<P>,
) {
    if block.luma {
        let taps = [LUMA16[block.fx & 3], LUMA16[block.fy & 3]];
        filter::<8, P>(plane, block, bit_depth, taps, (out, scratch));
    } else {
        let taps = [CHROMA16[block.fx & 7], CHROMA16[block.fy & 7]];
        filter::<4, P>(plane, block, bit_depth, taps, (out, scratch));
    }
}

/// Predict a block straight to 8-bit samples with the vector kernels:
/// from one list, or (`other` the intermediates of list 0, rows `w`
/// apart) the average with it. `false`, with nothing written, when the
/// kernels do not take the block; the caller then interpolates it.
#[inline(always)]
pub fn predict_px<P: Pixel>(
    plane: &Plane<'_, P>,
    block: Block,
    (dst, stride): (&mut [P], usize),
    (other, weights): (Option<&[i16]>, Option<Weights>),
    bit_depth: u32,
    scratch: &mut Scratch<P>,
) -> bool {
    if P::bytes(&[]).is_none() {
        return deep(
            plane,
            block,
            (dst, stride),
            (other, weights),
            bit_depth,
            scratch,
        );
    }
    let Some(dst) = P::bytes_mut(dst) else {
        return false;
    };
    let dst = match (other, weights) {
        (Some(o), None) => Dst::Bi(dst, o),
        (None, None) => Dst::Uni(dst),
        (Some(o), Some(w)) => Dst::BiW(dst, o, w),
        (None, Some(w)) => Dst::UniW(dst, w),
    };
    if block.luma {
        let taps = [LUMA16[block.fx & 3], LUMA16[block.fy & 3]];
        by_kernel::<8, P>(plane, block, taps, dst, stride, scratch)
    } else {
        let taps = [CHROMA16[block.fx & 7], CHROMA16[block.fy & 7]];
        by_kernel::<4, P>(plane, block, taps, dst, stride, scratch)
    }
}

/// [`predict_px`] for samples of 9 to 12 bits.
#[inline(always)]
fn deep<P: Pixel>(
    plane: &Plane<'_, P>,
    block: Block,
    (dst, stride): (&mut [P], usize),
    fuse: (Option<&[i16]>, Option<Weights>),
    bit_depth: u32,
    scratch: &mut Scratch<P>,
) -> bool {
    if block.luma {
        deep_t::<8, P>(plane, block, (dst, stride), fuse, bit_depth, scratch)
    } else {
        deep_t::<4, P>(plane, block, (dst, stride), fuse, bit_depth, scratch)
    }
}

#[inline(always)]
fn deep_t<const T: usize, P: Pixel>(
    plane: &Plane<'_, P>,
    block: Block,
    (dst, stride): (&mut [P], usize),
    (other, weights): (Option<&[i16]>, Option<Weights>),
    bit_depth: u32,
    scratch: &mut Scratch<P>,
) -> bool {
    let Some(px) = P::words_mut(dst) else {
        return false;
    };
    let (w, h) = (block.w.min(64), block.h.min(64));
    let Some(spot) = place::<T, P>(plane, block) else {
        return false;
    };
    let half = (T / 2 - 1) as isize;
    let (sw, sh) = (w + T - 1, h + T - 1);
    let Scratch { win, deep, .. } = scratch;
    // The kernel reads the reference itself, rows held within those of
    // the plane, unless the block reaches out on the left or the right.
    let (src, at, sstride, clamp): (&[P], usize, usize, _) = match spot {
        Place::Inside(at) => (plane.samples, at, plane.width, None),
        Place::Rows(at, top, rows) => (plane.samples, at, plane.width, Some((top, rows))),
        Place::Pad => {
            pad(
                plane,
                (block.x - half, block.y - half),
                sw,
                &mut win[..sw * sh],
            );
            (&*win, 0, sw, None)
        }
    };
    let Some(words) = P::words(src) else {
        return false;
    };
    let taps = Taps {
        across: (block.fx != 0).then(|| taps8::<T>(block.fx)),
        down: (block.fy != 0).then(|| taps8::<T>(block.fy)),
        len: T,
    };
    let s = cutlass_simd::hevc::mc16::Src {
        src: words,
        at,
        stride: sstride,
        clamp,
    };
    let out = match (other, weights) {
        (Some(o), None) => cutlass_simd::hevc::mc16::Dst::Bi(px, o),
        (None, None) => cutlass_simd::hevc::mc16::Dst::Uni(px),
        (Some(o), Some(w)) => cutlass_simd::hevc::mc16::Dst::BiW(px, o, w),
        (None, Some(w)) => cutlass_simd::hevc::mc16::Dst::UniW(px, w),
    };
    cutlass_simd::hevc::mc16::predict(out, stride, &s, (w, h), &taps, bit_depth, deep)
}

/// A block's source window: samples, where it starts in them, its stride,
/// and the rows to hold it within ([`Src::clamp`]).
type Window<'s, P> = (&'s [P], usize, usize, Option<(isize, usize)>);

/// Where a block's source window comes from.
#[derive(Clone, Copy)]
enum Place {
    /// Inside the rows held: its first sample is at this offset.
    Inside(usize),
    /// Inside across, reaching above or below the rows held: the column
    /// of its first sample, its first row, and the rows held (see
    /// [`Src::clamp`]).
    Rows(usize, isize, usize),
    /// Reaching outside on the left or the right: copied with its edges
    /// repeated.
    Pad,
}

/// Where the window of `block` comes from, for `T` taps; `None` for a
/// reference with no rows.
#[inline(always)]
fn place<const T: usize, P>(plane: &Plane<'_, P>, block: Block) -> Option<Place> {
    let (w, h) = (block.w.min(64), block.h.min(64));
    let half = (T / 2 - 1) as isize;
    let (sw, sh) = (w + T - 1, h + T - 1);
    let pw = plane.width as isize;
    let (x0, y0) = (block.x - half, block.y - half);
    let rows = plane.rows.min(plane.height);
    if pw == 0 || rows == 0 || plane.samples.len() < plane.width * rows {
        return None;
    }
    let across = x0 >= 0 && x0 + sw as isize <= pw;
    Some(if across && y0 >= 0 && y0 + sh as isize <= rows as isize {
        Place::Inside(y0 as usize * plane.width + x0 as usize)
    } else if across && x0 as usize + w + 24 <= plane.width {
        // The kernels' loads run `w + 24` samples into a row: past its end
        // but for the last row.
        Place::Rows(x0 as usize, y0, rows)
    } else {
        Place::Pad
    })
}

/// The source window of a block for the kernels: the reference itself
/// (with its rows held within those of the plane when the block reaches
/// above or below them), or `win` filled with the window, edges repeated.
/// The samples, where the window starts, its stride, and the rows to hold
/// it within ([`Src::clamp`]).
#[inline(always)]
fn window<'s, const T: usize, P: Pixel>(
    plane: &'s Plane<'s, P>,
    block: Block,
    win: &'s mut [P],
) -> Option<Window<'s, P>> {
    Some(match place::<T, P>(plane, block)? {
        Place::Inside(at) => (plane.samples, at, plane.width, None),
        Place::Rows(at, top, rows) => (plane.samples, at, plane.width, Some((top, rows))),
        Place::Pad => {
            let (w, h) = (block.w.min(64), block.h.min(64));
            let half = (T / 2 - 1) as isize;
            let (sw, sh) = (w + T - 1, h + T - 1);
            pad(
                plane,
                (block.x - half, block.y - half),
                sw,
                &mut win[..sw * sh],
            );
            (&*win, 0, sw, None)
        }
    })
}

/// [`predict_px`] for `T` taps.
#[inline(always)]
fn by_kernel<const T: usize, P: Pixel>(
    plane: &Plane<'_, P>,
    block: Block,
    _: [[i16; T]; 2],
    dst: Dst<'_>,
    stride: usize,
    scratch: &mut Scratch<P>,
) -> bool {
    let (w, h) = (block.w.min(64), block.h.min(64));
    let Scratch { win, rows, .. } = scratch;
    let Some((src, at, sstride, clamp)) = window::<T, P>(plane, block, win) else {
        return false;
    };
    let Some(bytes) = P::bytes(src) else {
        return false;
    };
    let taps = Taps {
        across: (block.fx != 0).then(|| taps8::<T>(block.fx)),
        down: (block.fy != 0).then(|| taps8::<T>(block.fy)),
        len: T,
    };
    let s = Src {
        src: bytes,
        at,
        stride: sstride,
        clamp,
    };
    kernels::predict(dst, stride, &s, (w, h), &taps, rows)
}

/// The taps of fraction `f` as the kernels take them: 8 for luma
/// (quarters), 4 then zeros for chroma (eighths).
#[inline(always)]
fn taps8<const T: usize>(f: usize) -> [i8; 8] {
    if T == 8 { LUMA8[f & 3] } else { CHROMA8[f & 7] }
}

const LUMA8: [[i8; 8]; 4] = {
    let mut t = [[0i8; 8]; 4];
    let mut f = 0;
    while f < 4 {
        let mut k = 0;
        while k < 8 {
            t[f][k] = LUMA[f][k] as i8;
            k += 1;
        }
        f += 1;
    }
    t
};

const CHROMA8: [[i8; 8]; 8] = {
    let mut t = [[0i8; 8]; 8];
    let mut f = 0;
    while f < 8 {
        let mut k = 0;
        while k < 4 {
            t[f][k] = CHROMA[f][k] as i8;
            k += 1;
        }
        f += 1;
    }
    t
};

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
pub struct Scratch<P> {
    win: Vec<P>,
    tmp: Vec<i16>,
    rows: kernels::Tmp,
    deep: cutlass_simd::hevc::mc16::Tmp,
}

impl<P: Pixel> Default for Scratch<P> {
    fn default() -> Self {
        Self {
            // Slack past the window for the 16-sample loads of the kernels.
            win: vec![P::default(); WINDOW * WINDOW + 32],
            tmp: vec![0; WINDOW * 64],
            rows: kernels::Tmp::default(),
            deep: cutlass_simd::hevc::mc16::Tmp::default(),
        }
    }
}

/// Largest window a 64×64 block with an 8-tap filter reads, per edge.
const WINDOW: usize = 64 + 7;

#[inline(always)]
fn filter<const T: usize, P: Pixel>(
    plane: &Plane<'_, P>,
    block: Block,
    bit_depth: u32,
    [cx, cy]: [[i16; T]; 2],
    (out, scratch): (&mut [i16], &mut Scratch<P>),
) {
    let Block {
        x, y, fx, fy, w, h, ..
    } = block;
    let (w, h) = (w.min(64), h.min(64));
    let half = (T / 2 - 1) as isize;
    let sw = w + T - 1;
    let sh = h + T - 1;
    let (x0, y0) = (x - half, y - half);
    let out = &mut out[..w * h];
    let Scratch {
        win,
        tmp,
        rows,
        deep,
    } = scratch;
    let Some(spot) = place::<T, P>(plane, block) else {
        out.fill(0);
        return;
    };
    let taps = Taps {
        across: (fx != 0).then(|| taps8::<T>(fx)),
        down: (fy != 0).then(|| taps8::<T>(fy)),
        len: T,
    };
    // The kernels read the reference itself unless the block reaches out
    // on the left or the right.
    if let (Some(bytes), Place::Inside(_) | Place::Rows(..)) = (P::bytes(plane.samples), spot) {
        let (at, clamp) = match spot {
            Place::Inside(at) => (at, None),
            Place::Rows(at, top, held) => (at, Some((top, held))),
            Place::Pad => (0, None),
        };
        let s = Src {
            src: bytes,
            at,
            stride: plane.width,
            clamp,
        };
        if kernels::put(out, w, &s, (w, h), &taps, rows) {
            return;
        }
    }
    // A window copied with its edges repeated.
    let (src, stride, base): (&[P], usize, usize) = match spot {
        Place::Inside(at) => (plane.samples, plane.width, at),
        _ => {
            pad(plane, (x0, y0), sw, &mut win[..sw * sh]);
            (&*win, sw, 0)
        }
    };
    if let Some(bytes) = P::bytes(src) {
        let s = Src {
            src: bytes,
            at: base,
            stride,
            clamp: None,
        };
        if kernels::put(out, w, &s, (w, h), &taps, rows) {
            return;
        }
    }
    if let Some(words) = P::words(src) {
        let s = cutlass_simd::hevc::mc16::Src {
            src: words,
            at: base,
            stride,
            clamp: None,
        };
        if cutlass_simd::hevc::mc16::put(out, w, &s, (w, h), &taps, bit_depth, deep) {
            return;
        }
    }
    let args = (&src[base..], stride, bit_depth, [cx, cy], (fx, fy), (w, h));
    if P::bytes(&[]).is_some() {
        scalar::<T, P>(args, out, tmp);
    } else {
        // Deeper samples always take this path: inline, so that it is
        // compiled with the caller, for AVX2 under `wide::wide`.
        loops::<T, P>(args, out, tmp);
    }
}

/// The source, stride, bit depth, taps, fractions and size of
/// [`loops`].
type Args<'s, const T: usize, P> = (
    &'s [P],
    usize,
    u32,
    [[i16; T]; 2],
    (usize, usize),
    (usize, usize),
);

/// [`loops`] for 8-bit samples on a CPU without the vector kernels. Out of
/// line, so the hot path stays small.
#[cold]
#[inline(never)]
fn scalar<const T: usize, P: Pixel>(args: Args<'_, T, P>, out: &mut [i16], tmp: &mut [i16]) {
    loops::<T, P>(args, out, tmp);
}

/// The interpolation as plain loops over rows of fixed width, which the
/// compiler turns into vector code.
#[inline(always)]
fn loops<const T: usize, P: Pixel>(
    (src, stride, bit_depth, [cx, cy], (fx, fy), (w, h)): Args<'_, T, P>,
    out: &mut [i16],
    tmp: &mut [i16],
) {
    let shift1 = bit_depth.saturating_sub(8).min(4);
    let shift3 = 14u32.saturating_sub(bit_depth).max(2);
    // Eight-bit samples times the taps fit 16 bits (at most 88 · 255).
    let k = Kernel {
        c: [cx, cy],
        mode: u8::from(fx != 0) | u8::from(fy != 0) << 1,
        narrow: bit_depth == 8,
        shift1,
        shift3,
    };
    // Widths of luma and chroma prediction blocks: powers of two, and
    // three quarters of one (asymmetric partitions, 4:2:0 of 6 or 12).
    macro_rules! go {
        ($w:literal, $at:expr) => {
            k.block::<$w, P>(&src[$at..], stride, &mut out[$at..], w, h, tmp)
        };
    }
    match w {
        64 => go!(64, 0),
        48 => {
            go!(32, 0);
            go!(16, 32);
        }
        32 => go!(32, 0),
        24 => {
            go!(16, 0);
            go!(8, 16);
        }
        16 => go!(16, 0),
        12 => {
            go!(8, 0);
            go!(4, 8);
        }
        8 => go!(8, 0),
        6 => {
            go!(4, 0);
            go!(2, 4);
        }
        4 => go!(4, 0),
        _ => {
            for at in (0..w).step_by(2) {
                go!(2, at);
            }
        }
    }
}

/// One interpolation: the taps across and down, which of them apply
/// (bit 0 across, bit 1 down), and the shifts of clause 8.5.3.3.3.
struct Kernel<const T: usize> {
    c: [[i16; T]; 2],
    mode: u8,
    narrow: bool,
    shift1: u32,
    shift3: u32,
}

impl<const T: usize> Kernel<T> {
    /// A block `W` wide and `h` high: `src` from the window's first
    /// sample (rows `stride` apart), `out` rows `ostride` apart.
    #[inline(always)]
    fn block<const W: usize, P: Pixel>(
        &self,
        src: &[P],
        stride: usize,
        out: &mut [i16],
        ostride: usize,
        h: usize,
        tmp: &mut [i16],
    ) {
        let h0 = T / 2 - 1;
        let row = |j: usize, from: usize, len: usize| -> &[P] {
            src.get(j * stride + from..j * stride + from + len)
                .unwrap_or(&[])
        };
        match self.mode {
            0 => {
                for j in 0..h {
                    let (Some(s), Some(o)) = (
                        row(j + h0, h0, W).first_chunk::<W>(),
                        out.get_mut(j * ostride..)
                            .and_then(|o| o.first_chunk_mut::<W>()),
                    ) else {
                        return;
                    };
                    for l in 0..W {
                        o[l] = (s[l].i() << self.shift3) as i16;
                    }
                }
            }
            1 => {
                for j in 0..h {
                    let s = row(j + h0, 0, W + T - 1);
                    let Some(o) = out
                        .get_mut(j * ostride..)
                        .and_then(|o| o.first_chunk_mut::<W>())
                    else {
                        return;
                    };
                    self.across::<W, P>(s, o);
                }
            }
            2 => {
                for j in 0..h {
                    let lines: [&[P]; T] = std::array::from_fn(|k| row(j + k, h0, W));
                    let Some(o) = out
                        .get_mut(j * ostride..)
                        .and_then(|o| o.first_chunk_mut::<W>())
                    else {
                        return;
                    };
                    self.down::<W, P>(&lines, o);
                }
            }
            _ => {
                let rows = h + T - 1;
                let Some(tmp) = tmp.get_mut(..rows * W) else {
                    return;
                };
                for (j, t) in tmp.as_chunks_mut::<W>().0.iter_mut().enumerate() {
                    self.across::<W, P>(row(j, 0, W + T - 1), t);
                }
                let tmp = tmp.as_chunks::<W>().0;
                for j in 0..h {
                    let Some(o) = out
                        .get_mut(j * ostride..)
                        .and_then(|o| o.first_chunk_mut::<W>())
                    else {
                        return;
                    };
                    let mut acc = [0i32; W];
                    for (k, &c) in self.c[1].iter().enumerate() {
                        let ck = i32::from(c);
                        let Some(t) = tmp.get(j + k) else {
                            return;
                        };
                        for l in 0..W {
                            acc[l] += ck * i32::from(t[l]);
                        }
                    }
                    for l in 0..W {
                        o[l] = (acc[l] >> 6) as i16;
                    }
                }
            }
        }
    }

    /// The taps across one row: `s` holds `W + T - 1` samples.
    #[inline(always)]
    fn across<const W: usize, P: Pixel>(&self, s: &[P], o: &mut [i16; W]) {
        let c = &self.c[0];
        if s.len() < W + T - 1 {
            return;
        }
        if self.narrow {
            let mut acc = [0i16; W];
            for k in 0..T {
                let ck = c[k];
                let Some(s) = s[k..].first_chunk::<W>() else {
                    return;
                };
                for l in 0..W {
                    acc[l] = acc[l].wrapping_add(ck.wrapping_mul(s[l].i() as i16));
                }
            }
            *o = acc;
        } else {
            let mut acc = [0i32; W];
            for k in 0..T {
                let ck = i32::from(c[k]);
                let Some(s) = s[k..].first_chunk::<W>() else {
                    return;
                };
                for l in 0..W {
                    acc[l] += ck * s[l].i();
                }
            }
            for l in 0..W {
                o[l] = (acc[l] >> self.shift1) as i16;
            }
        }
    }

    /// The taps down `T` rows of `W` samples.
    #[inline(always)]
    fn down<const W: usize, P: Pixel>(&self, lines: &[&[P]; T], o: &mut [i16; W]) {
        let c = &self.c[1];
        if self.narrow {
            let mut acc = [0i16; W];
            for k in 0..T {
                let ck = c[k];
                let Some(s) = lines[k].first_chunk::<W>() else {
                    return;
                };
                for l in 0..W {
                    acc[l] = acc[l].wrapping_add(ck.wrapping_mul(s[l].i() as i16));
                }
            }
            *o = acc;
        } else {
            let mut acc = [0i32; W];
            for k in 0..T {
                let ck = i32::from(c[k]);
                let Some(s) = lines[k].first_chunk::<W>() else {
                    return;
                };
                for l in 0..W {
                    acc[l] += ck * s[l].i();
                }
            }
            for l in 0..W {
                o[l] = (acc[l] >> self.shift1) as i16;
            }
        }
    }
}

/// Copy the `sw`×`sh` window at (`x0`, `y0`) of `plane` into `win`, the
/// samples outside the plane repeating its nearest edge.
#[inline(always)]
fn pad<P: Pixel>(plane: &Plane<'_, P>, (x0, y0): (isize, isize), sw: usize, win: &mut [P]) {
    // Rows past those held repeat the last held: a damaged stream only.
    let (pw, ph) = (
        plane.width as isize,
        (plane.height.min(plane.rows)) as isize,
    );
    // Columns of the window inside the plane: `a..b`.
    let a = (-x0).clamp(0, sw as isize) as usize;
    let b = (pw - x0).clamp(0, sw as isize) as usize;
    for (j, dst) in win.chunks_exact_mut(sw).enumerate() {
        let sy = (y0 + j as isize).clamp(0, ph - 1) as usize * plane.width;
        let line = &plane.samples[sy..sy + plane.width];
        if a < b {
            let from = (x0 + a as isize) as usize;
            copy_short(&mut dst[a..b], &line[from..from + (b - a)]);
            if a > 0 {
                dst[..a].fill(line[0]);
            }
            if b < sw {
                dst[b..].fill(line[plane.width - 1]);
            }
        } else {
            // Wholly left or right of the plane: its nearest edge sample.
            let v = if x0 >= pw {
                line[plane.width - 1]
            } else {
                line[0]
            };
            dst.fill(v);
        }
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
pub fn store<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    w: usize,
    h: usize,
    bit_depth: u32,
    pred: [Option<&[i16]>; 2],
    weights: Option<[Wp; 2]>,
) {
    if weights.is_none()
        && let Some(bytes) = P::bytes_mut(dst)
    {
        let done = match pred {
            [Some(a), Some(b)] => kernels::bi(bytes, stride, (a, b), w, (w, h)),
            [Some(a), None] | [None, Some(a)] => kernels::uni(bytes, stride, a, w, (w, h)),
            [None, None] => true,
        };
        if done {
            return;
        }
    }
    if weights.is_none()
        && let Some(words) = P::words_mut(dst)
    {
        let done = match pred {
            [Some(a), b @ Some(_)] | [Some(a), b @ None] | [b @ None, Some(a)] => {
                cutlass_simd::hevc::mc16::store(words, stride, (a, b), w, (w, h), bit_depth)
            }
            [None, None] => true,
        };
        if done {
            return;
        }
    }
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
                    *out = P::of(v.clamp(0, max));
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
                    *out = P::of(v.clamp(0, max));
                }
            }
        }
        ([Some(a), None] | [None, Some(a)], None) => {
            let add = 1i32 << (shift1 - 1);
            for (row, a) in rows.zip(a.chunks_exact(w)) {
                for (out, &a) in row[..w].iter_mut().zip(a) {
                    let v = (i32::from(a) + add) >> shift1;
                    *out = P::of(v.clamp(0, max));
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
                    *out = P::of(v.clamp(0, max));
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
        let samples: Vec<u16> = (0..64).collect();
        let plane = Plane {
            samples: &samples,
            width: 8,
            height: 8,
            rows: 8,
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
            rows: 4,
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

#[cfg(test)]
mod speed {
    use super::{Block, Plane, Scratch, interpolate, store};

    /// Nanoseconds per block of interpolation and storing, by size and
    /// fraction (run by hand: `--ignored --nocapture`).
    #[test]
    #[ignore = "timing"]
    fn mc_speed() {
        let (w0, h0) = (1920usize, 1080usize);
        let samples: Vec<u8> = (0..w0 * h0).map(|i| (i * 7 % 251) as u8).collect();
        let plane = Plane {
            samples: &samples,
            width: w0,
            height: h0,
            rows: h0,
        };
        let mut scratch = Scratch::<u8>::default();
        let mut out = vec![0i16; 64 * 64];
        let mut dst = vec![0u8; w0 * 64];
        for (w, h) in [(8usize, 8usize), (16, 16), (32, 32), (4, 4)] {
            for (fx, fy) in [(0usize, 0usize), (1, 0), (0, 2), (1, 3)] {
                let n = 200_000 / (w * h / 16);
                let t = std::time::Instant::now();
                for i in 0..n {
                    // Raster order over a strip, as a picture is decoded.
                    let per_row = 1800 / w;
                    let block = Block {
                        x: ((i % per_row) * w + i % 3) as isize,
                        y: (((i / per_row) * h) % 960 + i % 5) as isize,
                        fx,
                        fy,
                        w,
                        h,
                        luma: true,
                    };
                    cutlass_simd::wide::wide(
                        #[inline(always)]
                        || {
                            interpolate(&plane, block, 8, &mut out, &mut scratch);
                            store(&mut dst, w0, w, h, 8, [Some(&out[..w * h]), None], None);
                        },
                    );
                }
                let ns = t.elapsed().as_nanos() as f64 / n as f64;
                println!(
                    "{w}x{h} f({fx},{fy}): {ns:.0} ns/block, {:.2} ns/sample",
                    ns / (w * h) as f64
                );
            }
        }
    }
}

/// `dst.copy_from_slice(src)` for rows of at most 127 samples, as copies
/// of fixed sizes the compiler writes inline: a call to `memcpy` for a few
/// dozen bytes costs more than the copy.
#[inline(always)]
fn copy_short<P: Copy>(dst: &mut [P], src: &[P]) {
    let n = dst.len().min(src.len());
    let mut at = 0;
    macro_rules! part {
        ($k:literal) => {
            if n & $k != 0 {
                if let (Some(d), Some(s)) = (
                    dst.get_mut(at..).and_then(|d| d.first_chunk_mut::<$k>()),
                    src.get(at..).and_then(|s| s.first_chunk::<$k>()),
                ) {
                    *d = *s;
                }
                at += $k;
            }
        };
    }
    part!(64);
    part!(32);
    part!(16);
    part!(8);
    part!(4);
    part!(2);
    part!(1);
    if n > 127 {
        dst[at..n].copy_from_slice(&src[at..n]);
    }
}
