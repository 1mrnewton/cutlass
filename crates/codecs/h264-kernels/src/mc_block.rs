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

/// Where a chroma block of both planes is read: the Cb and Cr planes,
/// whose rows are `stride` apart, at the same sample `at`, at eighth
/// offsets `frac` (each 0 to 7).
#[derive(Clone, Copy)]
pub struct ChromaSource<'a> {
    pub cb: &'a [u8],
    pub cr: &'a [u8],
    pub stride: usize,
    pub at: (usize, usize),
    pub frac: (u8, u8),
}

impl<'a> ChromaSource<'a> {
    /// The block of `size` in one plane, Cb when `cb` is set.
    pub fn plane(&self, cb: bool, size: (usize, usize)) -> Block<'a> {
        Block {
            src: if cb { self.cb } else { self.cr },
            stride: self.stride,
            at: self.at,
            size,
        }
    }
}

/// Weights for mixing two predictions, clause 8.4.2.3: each sample is
/// `((a·w0 + b·w1 + 2^(shift−1)) >> shift) + offset`, clipped.
#[derive(Clone, Copy, Debug)]
pub struct BiWeight {
    pub w0: i32,
    pub w1: i32,
    pub shift: i32,
    pub offset: i32,
}

impl BiWeight {
    /// Whether the vector kernels' 16-bit weights and shift can hold it.
    pub(crate) fn fits(self) -> bool {
        let small = |weight: i32| (-256..=256).contains(&weight);
        small(self.w0) && small(self.w1) && (1..=15).contains(&self.shift)
    }
}

/// Weights for one prediction, clause 8.4.2.3.2: each sample is
/// `((p·weight + 2^(shift−1)) >> shift) + offset`, or `p·weight + offset`
/// when `shift` is 0, clipped.
#[derive(Clone, Copy, Debug)]
pub struct UniWeight {
    pub weight: i32,
    pub offset: i32,
    pub shift: i32,
}

impl UniWeight {
    /// Whether a weighted sample and its offset fit in 16 bits: the range
    /// clause 7.4.3.2 gives 8-bit weights and offsets, and a little more.
    pub(crate) fn fits(self) -> bool {
        (-128..=128).contains(&self.weight)
            && (-255..=255).contains(&self.offset)
            && (0..=7).contains(&self.shift)
    }
}

/// Whether a block of `size` weighted in place fits `dst`, with rows
/// `dst_stride` apart: widths a multiple of 4 up to 16.
pub(crate) fn scale_fits(dst: &[u8], dst_stride: usize, size: (usize, usize)) -> bool {
    let (w, h) = size;
    w > 0
        && w <= 16
        && w % 4 == 0
        && h > 0
        && dst_stride >= w
        && (h - 1)
            .checked_mul(dst_stride)
            .and_then(|start| start.checked_add(w))
            .is_some_and(|end| end <= dst.len())
}

/// Whether a block of `size` mixed by `weight` into the one at `dst`,
/// rows `dst_stride` apart, from `b`, rows `size.0` apart, fits both and
/// the kernels' weights: widths a multiple of 4 up to 16.
pub(crate) fn onto_fits(
    dst: &[u8],
    dst_stride: usize,
    b: &[u8],
    size: (usize, usize),
    weight: BiWeight,
) -> bool {
    let (w, h) = size;
    weight.fits() && scale_fits(dst, dst_stride, size) && b.len() >= w * h
}

/// One mixed sample, for the widths the row kernels leave over.
#[inline(always)]
pub(crate) fn weight_one(a: u8, b: u8, weight: BiWeight) -> u8 {
    let sum = i32::from(a) * weight.w0 + i32::from(b) * weight.w1 + (1 << (weight.shift - 1));
    ((sum >> weight.shift) + weight.offset).clamp(0, 255) as u8
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
/// `avg16`, `avg8`, `chroma8`, `weight16` and `weight8`. The visibility and
/// the attribute, if any, go on each.
macro_rules! block_kernels {
    ($visibility:vis, $(#[$feature:meta])*) => {
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

/// The whole-block predictors over whichever row kernels are in scope
/// where this is used, as for [`block_kernels`]. The NEON build has its own
/// in `neon_qpel`; this one is the plain twin.
macro_rules! predict_kernels {
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
            // Every row reads `w + 1` samples of its own source row and the
            // one below, so the last reaches row `y + h`. A block past the
            // source or `dst` is refused, as the vector kernels refuse it,
            // and the caller predicts it from a padded copy instead.
            if h == 0 || dst_stride < w || (h - 1) * dst_stride + w > dst.len() {
                return false;
            }
            let reach = (y + h)
                .checked_mul(stride)
                .and_then(|row| row.checked_add(x + w + 1));
            if reach.is_none_or(|end| end > src.len()) {
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

        /// The rounded average of two blocks of `size`, whose rows are
        /// `size.0` apart, into `dst` with rows `dst_stride` apart.
        $(#[$feature])*
        $visibility fn avg_block(
            dst: &mut [u8],
            dst_stride: usize,
            a: &[u8],
            b: &[u8],
            size: (usize, usize),
        ) -> bool {
            let (w, h) = size;
            if w == 0 || h == 0 || a.len() < w * h || b.len() < w * h {
                return false;
            }
            if dst.len() < (h - 1) * dst_stride + w {
                return false;
            }
            for row in 0..h {
                let out = &mut dst[row * dst_stride..row * dst_stride + w];
                avg_row(out, &a[row * w..row * w + w], &b[row * w..row * w + w]);
            }
            true
        }

        /// Two blocks mixed by `weight`, laid out as for `avg_block`.
        /// `false` means the weights are out of the kernels' range, or a
        /// block does not fit, and nothing was done.
        $(#[$feature])*
        $visibility fn weight_block(
            dst: &mut [u8],
            dst_stride: usize,
            a: &[u8],
            b: &[u8],
            size: (usize, usize),
            weight: $crate::mc_block::BiWeight,
        ) -> bool {
            use $crate::mc_block::{at8, at16, weight_one};
            let (w, h) = size;
            if !weight.fits() || w == 0 || h == 0 || a.len() < w * h || b.len() < w * h {
                return false;
            }
            if dst.len() < (h - 1) * dst_stride + w {
                return false;
            }
            for row in 0..h {
                let out = &mut dst[row * dst_stride..row * dst_stride + w];
                let (a, b) = (&a[row * w..row * w + w], &b[row * w..row * w + w]);
                let mut done = 0;
                while done + 16 <= w {
                    let mixed = weight16(at16(a, done), at16(b, done), weight);
                    out[done..done + 16].copy_from_slice(&mixed);
                    done += 16;
                }
                if done + 8 <= w {
                    let mixed = weight8(at8(a, done), at8(b, done), weight);
                    out[done..done + 8].copy_from_slice(&mixed);
                    done += 8;
                }
                for ((sample, &left), &right) in out[done..].iter_mut().zip(&a[done..]).zip(&b[done..]) {
                    *sample = weight_one(left, right, weight);
                }
            }
            true
        }

    };
}
pub(crate) use predict_kernels;

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

/// A reference picture's 4:2:0 planes as [`crate::mc::predict_420`] reads
/// them: luma `width`×`height` samples with rows `stride` apart, and Cb and
/// Cr half as wide and high with rows `stride / 2` apart. The planes may
/// stop short of the picture's last rows, as a picture another thread is
/// still decoding does; a block is predicted only from rows they hold.
#[derive(Clone, Copy)]
pub struct Planes420<'a> {
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    stride: usize,
    size: (i64, i64),
}

impl<'a> Planes420<'a> {
    /// `None` when the luma is wider than `stride`, `stride` is odd, or the
    /// Cb and Cr planes differ in length.
    #[inline]
    pub fn new(
        (y, u, v): (&'a [u8], &'a [u8], &'a [u8]),
        stride: usize,
        (width, height): (usize, usize),
    ) -> Option<Self> {
        if width > stride || !stride.is_multiple_of(2) || u.len() != v.len() {
            return None;
        }
        let size = (i64::try_from(width).ok()?, i64::try_from(height).ok()?);
        Some(Self {
            y,
            u,
            v,
            stride,
            size,
        })
    }

    pub(crate) fn y(&self) -> &'a [u8] {
        self.y
    }

    pub(crate) fn cb(&self) -> &'a [u8] {
        self.u
    }

    pub(crate) fn cr(&self) -> &'a [u8] {
        self.v
    }

    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    /// Where a luma block of `size` (8 or 16 wide, 1 to 16 high) whose
    /// first full sample is at `at`, at quarter offsets `frac`, is read
    /// from: the place of that sample in the luma plane. `None` unless
    /// every sample its six taps read is inside the picture, and every
    /// byte the NEON routine loads is in the plane: in each direction with
    /// an offset, two samples before the block and three after it, and
    /// along a row with one, `2 × width` bytes from two before it.
    #[inline]
    pub(crate) fn luma_from(
        &self,
        at: (i64, i64),
        size: (usize, usize),
        frac: (u8, u8),
    ) -> Option<usize> {
        let (w, h) = (size.0 as i64, size.1 as i64);
        let (left, right, span) = if frac.0 == 0 {
            (0, 0, w)
        } else {
            (2, 3, 2 * w)
        };
        let (above, below) = if frac.1 == 0 { (0, 0) } else { (2, 3) };
        if at.0 < left
            || at.1 < above
            || at.0 + w + right > self.size.0
            || at.1 + h + below > self.size.1
        {
            return None;
        }
        // In the picture, so both are at least 0 and fit a row and a plane.
        let (x, y) = (at.0 as usize, at.1 as usize);
        let last = (y + size.1 - 1 + below as usize)
            .checked_mul(self.stride)?
            .checked_add(x - left as usize + span as usize)?;
        if last > self.y.len() {
            return None;
        }
        Some(y * self.stride + x)
    }

    /// [`Self::luma_from`] for a chroma block of `size` (4 or 8 wide, an
    /// even height up to 8) at eighth offsets, `at` in the chroma planes:
    /// inside the picture with one more column and row where it has an
    /// offset that way, and each row the routine reads, nine bytes from the
    /// block's first sample and one row below its last, in both planes.
    #[inline]
    pub(crate) fn chroma_from(
        &self,
        at: (i64, i64),
        size: (usize, usize),
        frac: (u8, u8),
    ) -> Option<usize> {
        let (w, h) = (size.0 as i64, size.1 as i64);
        if at.0 < 0
            || at.1 < 0
            || at.0 + w + i64::from(frac.0 != 0) > self.size.0 / 2
            || at.1 + h + i64::from(frac.1 != 0) > self.size.1 / 2
        {
            return None;
        }
        let (x, y) = (at.0 as usize, at.1 as usize);
        let stride = self.stride / 2;
        let last = (y + size.1).checked_mul(stride)?.checked_add(x + 9)?;
        if last > self.u.len() {
            return None;
        }
        Some(y * stride + x)
    }
}

/// The 4:2:0 picture [`crate::mc::predict_420`] writes: as
/// [`Planes420`], and holding every row of the picture.
pub struct Planes420Mut<'a> {
    y: &'a mut [u8],
    u: &'a mut [u8],
    v: &'a mut [u8],
    stride: usize,
    size: (usize, usize),
}

impl<'a> Planes420Mut<'a> {
    /// `None` when the luma is wider than `stride`, `stride` is odd, or a
    /// plane is shorter than the picture.
    #[inline]
    pub fn new(
        (y, u, v): (&'a mut [u8], &'a mut [u8], &'a mut [u8]),
        stride: usize,
        (width, height): (usize, usize),
    ) -> Option<Self> {
        let chroma = (stride / 2).checked_mul(height / 2)?;
        if width > stride
            || !stride.is_multiple_of(2)
            || y.len() < stride.checked_mul(height)?
            || u.len() < chroma
            || v.len() < chroma
        {
            return None;
        }
        Some(Self {
            y,
            u,
            v,
            stride,
            size: (width, height),
        })
    }

    /// Whether the luma block of `size` at `at` lies in the picture, and so
    /// do the chroma blocks at half each.
    #[inline]
    pub(crate) fn holds(&self, at: (usize, usize), size: (usize, usize)) -> bool {
        at.0.checked_add(size.0)
            .is_some_and(|right| right <= self.size.0)
            && at
                .1
                .checked_add(size.1)
                .is_some_and(|below| below <= self.size.1)
    }

    pub(crate) fn planes(&mut self) -> (&mut [u8], &mut [u8], &mut [u8], usize) {
        (&mut *self.y, &mut *self.u, &mut *self.v, self.stride)
    }
}
