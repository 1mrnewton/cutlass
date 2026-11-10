//! 8-bit inter prediction through the vector kernels.
//!
//! Each function here does what its namesake in `recon` does for any sample
//! width, for the block sizes the kernels cover, and says `false` for the
//! rest so the caller runs its own loop. All of it is safe indexing: a
//! kernel only ever gets arrays cut from the slices here.

use cutlass_h264_kernels::mc;

const ZERO16: [u8; 16] = [0; 16];
const ZERO8: [u8; 8] = [0; 8];

/// The 16 samples of `samples` from `start`.
fn at16(samples: &[u8], start: usize) -> &[u8; 16] {
    samples[start..start + 16].try_into().unwrap_or(&ZERO16)
}

/// The 8 samples of `samples` from `start`.
fn at8(samples: &[u8], start: usize) -> &[u8; 8] {
    samples[start..start + 8].try_into().unwrap_or(&ZERO8)
}

/// Six lines of a block's row for the six-tap filter, the first starting at
/// `start` and each next one `step` further on.
fn taps16(samples: &[u8], start: usize, step: usize) -> [&[u8; 16]; 6] {
    core::array::from_fn(|tap| at16(samples, start + tap * step))
}

fn taps8(samples: &[u8], start: usize, step: usize) -> [&[u8; 8]; 6] {
    core::array::from_fn(|tap| at8(samples, start + tap * step))
}

/// A block of `size` whose top-left full sample is at `at` in `src`.
#[derive(Clone, Copy)]
pub(crate) struct Block<'a> {
    pub src: &'a [u8],
    pub stride: usize,
    pub at: (usize, usize),
    pub size: (usize, usize),
}

/// Horizontal half samples.
pub(crate) fn half_h(dst: &mut [u8], dst_stride: usize, block: Block<'_>) -> bool {
    let Block {
        src,
        stride,
        at: (x, y),
        size: (w, h),
    } = block;
    if w != 8 && w != 16 {
        return false;
    }
    for row in 0..h {
        let start = (y + row) * stride + x - 2;
        let out = &mut dst[row * dst_stride..row * dst_stride + w];
        if w == 16 {
            out.copy_from_slice(&mc::half16(taps16(src, start, 1)));
        } else {
            out.copy_from_slice(&mc::half8(taps8(src, start, 1)));
        }
    }
    true
}

/// Vertical half samples.
pub(crate) fn half_v(dst: &mut [u8], dst_stride: usize, block: Block<'_>) -> bool {
    let Block {
        src,
        stride,
        at: (x, y),
        size: (w, h),
    } = block;
    if w != 8 && w != 16 {
        return false;
    }
    for row in 0..h {
        let start = (y + row - 2) * stride + x;
        let out = &mut dst[row * dst_stride..row * dst_stride + w];
        if w == 16 {
            out.copy_from_slice(&mc::half16(taps16(src, start, stride)));
        } else {
            out.copy_from_slice(&mc::half8(taps8(src, start, stride)));
        }
    }
    true
}

/// Centre samples: the horizontal filter unrounded on the block's rows and
/// the five around them, then the vertical filter on those sums.
pub(crate) fn diag(dst: &mut [u8], dst_stride: usize, block: Block<'_>) -> bool {
    let Block {
        src,
        stride,
        at: (x, y),
        size: (w, h),
    } = block;
    if h > 16 {
        return false;
    }
    if w == 16 {
        let mut sums = [[0i16; 16]; 21];
        for (line, sum) in sums.iter_mut().enumerate().take(h + 5) {
            *sum = mc::raw16(taps16(src, (y + line - 2) * stride + x - 2, 1));
        }
        for row in 0..h {
            let taps = core::array::from_fn(|tap| &sums[row + tap]);
            dst[row * dst_stride..row * dst_stride + 16].copy_from_slice(&mc::centre16(taps));
        }
        return true;
    }
    if w == 8 {
        let mut sums = [[0i16; 8]; 21];
        for (line, sum) in sums.iter_mut().enumerate().take(h + 5) {
            *sum = mc::raw8(taps8(src, (y + line - 2) * stride + x - 2, 1));
        }
        for row in 0..h {
            let taps = core::array::from_fn(|tap| &sums[row + tap]);
            dst[row * dst_stride..row * dst_stride + 8].copy_from_slice(&mc::centre8(taps));
        }
        return true;
    }
    false
}

/// The rounded average of two rows.
pub(crate) fn avg_row(dst: &mut [u8], a: &[u8], b: &[u8]) -> bool {
    let n = dst.len();
    if a.len() < n || b.len() < n {
        return false;
    }
    let mut done = 0;
    while done + 16 <= n {
        dst[done..done + 16].copy_from_slice(&mc::avg16(at16(a, done), at16(b, done)));
        done += 16;
    }
    for ((sample, &left), &right) in dst[done..].iter_mut().zip(&a[done..n]).zip(&b[done..n]) {
        *sample = ((u16::from(left) + u16::from(right) + 1) >> 1) as u8;
    }
    true
}

/// Chroma samples at eighth-sample offsets `frac`. The block's last column
/// and row each have one more sample after them in `src`.
pub(crate) fn chroma(dst: &mut [u8], dst_stride: usize, block: Block<'_>, frac: (u8, u8)) -> bool {
    let Block {
        src,
        stride,
        at: (x, y),
        size: (w, h),
    } = block;
    let (xf, yf) = frac;
    if w != 4 && w != 8 {
        return false;
    }
    for row in 0..h {
        let start = (y + row) * stride + x;
        let out = &mut dst[row * dst_stride..row * dst_stride + w];
        if w == 8 {
            let top = nine::<9>(src, start);
            let bottom = nine::<9>(src, start + stride);
            out.copy_from_slice(&mc::chroma8(&top, &bottom, xf, yf));
        } else {
            // A block of 4 uses the first five samples; the rest stay zero
            // and only feed outputs that are dropped.
            let top = nine::<5>(src, start);
            let bottom = nine::<5>(src, start + stride);
            out.copy_from_slice(&mc::chroma8(&top, &bottom, xf, yf)[..4]);
        }
    }
    true
}

/// The `N` samples of `samples` from `start`, then zeros up to nine.
fn nine<const N: usize>(samples: &[u8], start: usize) -> [u8; 9] {
    let mut out = [0u8; 9];
    for (sample, &value) in out.iter_mut().zip(&samples[start..start + N]) {
        *sample = value;
    }
    out
}
