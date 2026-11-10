//! Whole-block inter prediction, written once and built twice: over the
//! NEON row kernels and over the plain ones. A block's rows are made in one
//! function, so the row kernels inline into it.
//!
//! Everything is safe indexing into the caller's slices. A block that does
//! not fit them panics; it never reads or writes outside them.

/// A block of `size` (width, height) whose top-left full sample is at `at`
/// (x, y) in `src`, a plane with rows `stride` apart.
#[derive(Clone, Copy)]
pub struct Block<'a> {
    pub src: &'a [u8],
    pub stride: usize,
    pub at: (usize, usize),
    pub size: (usize, usize),
}

const ZERO16: [u8; 16] = [0; 16];
const ZERO8: [u8; 8] = [0; 8];

/// The 16 samples of `samples` from `start`.
#[inline(always)]
pub(crate) fn at16(samples: &[u8], start: usize) -> &[u8; 16] {
    samples[start..start + 16].try_into().unwrap_or(&ZERO16)
}

/// The 8 samples of `samples` from `start`.
#[inline(always)]
pub(crate) fn at8(samples: &[u8], start: usize) -> &[u8; 8] {
    samples[start..start + 8].try_into().unwrap_or(&ZERO8)
}

/// The `N` samples of `samples` from `start`, then zeros up to nine.
#[inline(always)]
pub(crate) fn nine<const N: usize>(samples: &[u8], start: usize) -> [u8; 9] {
    let mut out = [0u8; 9];
    for (sample, &value) in out.iter_mut().zip(&samples[start..start + N]) {
        *sample = value;
    }
    out
}

/// One of the two sets of samples a quarter sample is the average of.
#[derive(Clone, Copy)]
pub(crate) enum Source {
    /// Full samples, this far right and down.
    Full(usize, usize),
    /// Horizontal half samples, this many rows down.
    HalfH(usize),
    /// Vertical half samples, this many columns right.
    HalfV(usize),
    /// Centre samples.
    Centre,
}

/// The two sources of the quarter sample at quarter offsets `frac`, clause
/// 8.4.2.2.1. `None` when the position is a full or half sample.
pub(crate) fn quarter_sources(frac: (u8, u8)) -> Option<(Source, Source)> {
    use Source::{Centre, Full, HalfH, HalfV};
    Some(match frac {
        (1, 0) => (Full(0, 0), HalfH(0)),
        (3, 0) => (HalfH(0), Full(1, 0)),
        (0, 1) => (Full(0, 0), HalfV(0)),
        (0, 3) => (HalfV(0), Full(0, 1)),
        (2, 1) => (HalfH(0), Centre),
        (1, 2) => (HalfV(0), Centre),
        (2, 3) => (Centre, HalfH(1)),
        (3, 2) => (Centre, HalfV(1)),
        (1, 1) => (HalfH(0), HalfV(0)),
        (3, 1) => (HalfH(0), HalfV(1)),
        (1, 3) => (HalfV(0), HalfH(1)),
        (3, 3) => (HalfV(1), HalfH(1)),
        _ => return None,
    })
}

/// The block functions over whichever row kernels are in scope where this
/// is used: `half16`, `half8`, `raw16`, `raw8`, `centre16`, `centre8`,
/// `avg16`, `avg8` and `chroma8`. The visibility and the attribute, if any,
/// go on each.
macro_rules! block_kernels {
    ($visibility:vis, $(#[$feature:meta])*) => {
        /// A luma block at quarter offsets `frac`, for blocks 8 or 16 wide
        /// and at most 16 high. `false` means nothing was done.
        $(#[$feature])*
        $visibility fn luma_block(
            dst: &mut [u8],
            dst_stride: usize,
            block: $crate::mc_block::Block<'_>,
            frac: (u8, u8),
        ) -> bool {
            let (w, h) = block.size;
            if h > 16 {
                return false;
            }
            match w {
                16 => luma16(dst, dst_stride, block, frac),
                8 => luma8(dst, dst_stride, block, frac),
                _ => return false,
            }
            true
        }

        $crate::mc_block::luma_rows!(
            $(#[$feature])* luma16, luma16_row, 16, at16, half16, raw16, centre16, avg16
        );
        $crate::mc_block::luma_rows!(
            $(#[$feature])* luma8, luma8_row, 8, at8, half8, raw8, centre8, avg8
        );

        /// A chroma block at eighth offsets `frac`, for blocks 4 or 8 wide.
        /// The block's last column and row each have one more sample after
        /// them in the source. `false` means nothing was done.
        $(#[$feature])*
        $visibility fn chroma_block(
            dst: &mut [u8],
            dst_stride: usize,
            block: $crate::mc_block::Block<'_>,
            frac: (u8, u8),
        ) -> bool {
            use $crate::mc_block::nine;
            let $crate::mc_block::Block {
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
                    out.copy_from_slice(&chroma8(&top, &bottom, xf, yf));
                } else {
                    // A block of 4 uses the first five samples; the rest
                    // stay zero and only feed outputs that are dropped.
                    let top = nine::<5>(src, start);
                    let bottom = nine::<5>(src, start + stride);
                    out.copy_from_slice(&chroma8(&top, &bottom, xf, yf)[..4]);
                }
            }
            true
        }

        /// The rounded average of `a` and `b` into `dst`, all of one length.
        $(#[$feature])*
        $visibility fn avg_row(dst: &mut [u8], a: &[u8], b: &[u8]) -> bool {
            use $crate::mc_block::{at8, at16};
            let n = dst.len();
            if a.len() < n || b.len() < n {
                return false;
            }
            let mut done = 0;
            while done + 16 <= n {
                dst[done..done + 16].copy_from_slice(&avg16(at16(a, done), at16(b, done)));
                done += 16;
            }
            if done + 8 <= n {
                dst[done..done + 8].copy_from_slice(&avg8(at8(a, done), at8(b, done)));
                done += 8;
            }
            let rest = dst[done..].iter_mut().zip(&a[done..n]).zip(&b[done..n]);
            for ((sample, &left), &right) in rest {
                *sample = ((u16::from(left) + u16::from(right) + 1) >> 1) as u8;
            }
            true
        }
    };
}
pub(crate) use block_kernels;

/// A luma block of one width at any quarter offset: each row is made from
/// its one or two sources straight into `dst`.
macro_rules! luma_rows {
    (
        $(#[$feature:meta])* $name:ident, $row:ident, $w:literal, $at:ident,
        $half:ident, $raw:ident, $centre:ident, $avg:ident
    ) => {
        $(#[$feature])*
        fn $name(
            dst: &mut [u8],
            dst_stride: usize,
            block: $crate::mc_block::Block<'_>,
            frac: (u8, u8),
        ) {
            use $crate::mc_block::{Source, $at};
            let $crate::mc_block::Block {
                src,
                stride,
                at: (x, y),
                size: (_, h),
            } = block;
            // A half or full sample has one source; a quarter sample is the
            // average of two.
            let (first, second) = match $crate::mc_block::quarter_sources(frac) {
                Some((first, second)) => (first, Some(second)),
                None => match frac {
                    (2, 0) => (Source::HalfH(0), None),
                    (0, 2) => (Source::HalfV(0), None),
                    (2, 2) => (Source::Centre, None),
                    _ => (Source::Full(0, 0), None),
                },
            };
            // Centre samples need the horizontal filter, unrounded, on the
            // block's rows and the five around them.
            let mut sums = [[0i16; $w]; 21];
            if matches!(first, Source::Centre) || matches!(second, Some(Source::Centre)) {
                for (line, sum) in sums.iter_mut().enumerate().take(h + 5) {
                    let start = (y + line - 2) * stride + x - 2;
                    let window: &[u8; $w + 5] = match src[start..start + $w + 5].try_into() {
                        Ok(window) => window,
                        Err(_) => return,
                    };
                    *sum = $raw(core::array::from_fn(|tap| $at(window, tap)));
                }
            }
            for row in 0..h {
                let out = &mut dst[row * dst_stride..row * dst_stride + $w];
                let made = $row(src, stride, (x, y + row), first, &sums, row);
                match second {
                    Some(second) => {
                        let other = $row(src, stride, (x, y + row), second, &sums, row);
                        out.copy_from_slice(&$avg(&made, &other));
                    }
                    None => out.copy_from_slice(&made),
                }
            }
        }

        /// One row of `source` for the block row whose first full sample
        /// is at `at`. `sums` are the first-pass sums and `row` the row's
        /// place in them.
        $(#[$feature])*
        #[inline]
        fn $row(
            src: &[u8],
            stride: usize,
            at: (usize, usize),
            source: $crate::mc_block::Source,
            sums: &[[i16; $w]; 21],
            row: usize,
        ) -> [u8; $w] {
            use $crate::mc_block::{Source, $at};
            let (x, y) = at;
            match source {
                Source::Full(dx, dy) => *$at(src, (y + dy) * stride + x + dx),
                Source::HalfH(dy) => {
                    // One window of the row holds all six taps.
                    let start = (y + dy) * stride + x - 2;
                    match <&[u8; $w + 5]>::try_from(&src[start..start + $w + 5]) {
                        Ok(window) => $half(core::array::from_fn(|tap| $at(window, tap))),
                        Err(_) => [0; $w],
                    }
                }
                Source::HalfV(dx) => {
                    // Six rows, each cut from what is left of the window.
                    let start = (y - 2) * stride + x + dx;
                    let window = &src[start..start + 5 * stride + $w];
                    let (r0, rest) = window.split_at(stride);
                    let (r1, rest) = rest.split_at(stride);
                    let (r2, rest) = rest.split_at(stride);
                    let (r3, rest) = rest.split_at(stride);
                    let (r4, r5) = rest.split_at(stride);
                    $half([
                        $at(r0, 0),
                        $at(r1, 0),
                        $at(r2, 0),
                        $at(r3, 0),
                        $at(r4, 0),
                        $at(r5, 0),
                    ])
                }
                Source::Centre => $centre([
                    &sums[row],
                    &sums[row + 1],
                    &sums[row + 2],
                    &sums[row + 3],
                    &sums[row + 4],
                    &sums[row + 5],
                ]),
            }
        }
    };
}
pub(crate) use luma_rows;
