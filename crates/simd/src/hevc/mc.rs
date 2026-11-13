//! Fractional sample interpolation of 8-bit HEVC pictures (clause
//! 8.5.3.3.3 of ITU-T H.265) into the 14-bit intermediates of the
//! weighted sample prediction, and the default weighted prediction of one
//! or two of them back to 8-bit samples (clause 8.5.3.3.4.2).
//!
//! An entry returns `false`, having written nothing, for a block it does
//! not take (its width, or a source too short for the loads): the caller
//! then predicts the block its own way. Every entry has a scalar twin in
//! the tests that the kernels are compared with.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(test)]
mod tests;

/// The reference samples of one block: `src[at]` is the first sample of
/// the window the filters read, `T / 2 - 1` rows above and columns left
/// of the block's first sample for a `T`-tap filter.
///
/// Row `j` of the window is row `j` from `at`, or, with `clamp`, the row
/// `top + j` of the plane, `top` being the window's first row relative to
/// the row of `at`, held within the plane's first `rows` rows: the window
/// of a block reaching above or below the picture, with no copy.
#[derive(Clone, Copy, Debug)]
pub struct Src<'a> {
    pub src: &'a [u8],
    pub at: usize,
    pub stride: usize,
    /// `(top, rows)`.
    pub clamp: Option<(isize, usize)>,
}

/// The filters of one block: taps across and down (`None`: the fraction
/// is 0), 8 taps for luma and 4 for chroma (the last four 0).
#[derive(Clone, Copy, Debug)]
pub struct Taps {
    pub across: Option<[i8; 8]>,
    pub down: Option<[i8; 8]>,
    /// 8 or 4.
    pub len: usize,
}

impl Src<'_> {
    /// Where row `j` of the window starts.
    #[inline]
    #[must_use]
    pub fn row(&self, j: usize) -> usize {
        match self.clamp {
            None => self.at + j * self.stride,
            Some((top, rows)) => {
                let r = (top + j as isize).clamp(0, rows.max(1) as isize - 1) as usize;
                self.at + r * self.stride
            }
        }
    }

    /// Whether every load of a `w`×`h` block stays inside `src`: the
    /// kernels read `w + 24` samples of each of `h + taps - 1` rows.
    fn holds(&self, w: usize, h: usize, taps: usize) -> bool {
        let last = match self.clamp {
            None => h + taps - 2,
            Some((top, rows)) => {
                (top + (h + taps - 2) as isize).clamp(0, rows.max(1) as isize - 1) as usize
            }
        };
        last.checked_mul(self.stride)
            .and_then(|r| r.checked_add(self.at))
            .and_then(|r| r.checked_add(w + 24))
            .is_some_and(|end| end <= self.src.len())
    }
}

/// Scratch rows the kernels keep between blocks: four vectors for each row
/// of a block's window, made on first use, where vector instructions may
/// run.
#[derive(Default)]
pub struct Tmp {
    #[cfg(target_arch = "x86_64")]
    rows: Vec<avx2::Strips>,
}

/// Whether the kernels of this module run on this CPU.
#[must_use]
pub fn accelerated() -> bool {
    crate::wide::has_avx2()
}

/// Where a block's prediction goes.
pub enum Dst<'a> {
    /// The 14-bit intermediates.
    Mid(&'a mut [i16]),
    /// Samples, `Clip1((x + 32) >> 6)` of the intermediates (one list).
    Uni(&'a mut [u8]),
    /// Samples, `Clip1((x + y + 64) >> 7)` with the intermediates `y` of
    /// the other list, whose rows are `w` apart.
    Bi(&'a mut [u8], &'a [i16]),
    /// Samples, explicitly weighted (equation 8-252):
    /// `Clip1(((x · w0 + 2^(log2wd − 1)) >> log2wd) + o0)`.
    UniW(&'a mut [u8], Weights),
    /// Samples, explicitly weighted with the intermediates `y` of list 0
    /// (rows `w` apart; equation 8-253):
    /// `Clip1((y · w0 + x · w1 + ((o0 + o1 + 1) << log2wd)) >> (log2wd + 1))`.
    BiW(&'a mut [u8], &'a [i16], Weights),
}

/// Explicit weights of a block: weights and offsets of lists 0 and 1, and
/// `log2WD`.
#[derive(Clone, Copy, Debug)]
pub struct Weights {
    pub w0: i16,
    pub w1: i16,
    pub o0: i32,
    pub o1: i32,
    pub log2wd: u32,
}

impl Dst<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Mid(d) => d.len(),
            Self::Uni(d) | Self::Bi(d, _) | Self::UniW(d, _) | Self::BiW(d, _, _) => d.len(),
        }
    }
}

/// The prediction of a `w`×`h` block into `dst` (rows `dst_stride`
/// apart). The intermediates are `x << 6` with no fraction, the sums of
/// the taps with one, and with both the vertical sums of the horizontal
/// ones shifted right by 6.
#[inline]
pub fn predict(
    dst: Dst<'_>,
    dst_stride: usize,
    src: &Src<'_>,
    (w, h): (usize, usize),
    taps: &Taps,
    tmp: &mut Tmp,
) -> bool {
    let fits = matches!(w, 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64)
        && (1..=64).contains(&h)
        && matches!(taps.len, 4 | 8)
        && dst_stride >= w
        && (h - 1) * dst_stride + w <= dst.len()
        && match &dst {
            Dst::Bi(_, other) => (h - 1) * w + w <= other.len(),
            Dst::BiW(_, other, wt) => {
                (h - 1) * w + w <= other.len() && (1..=14).contains(&wt.log2wd)
            }
            Dst::UniW(_, wt) => (1..=14).contains(&wt.log2wd),
            _ => true,
        }
        && src.holds(w, h, taps.len);
    if !fits || !accelerated() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `accelerated` checked at run time that the CPU has AVX2.
        #[allow(unsafe_code)]
        unsafe {
            avx2::predict(dst, dst_stride, src, (w, h), taps, &mut tmp.rows);
        }
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (dst, tmp);
        false
    }
}

/// [`predict`] into intermediates.
#[inline]
pub fn put(
    dst: &mut [i16],
    dst_stride: usize,
    src: &Src<'_>,
    (w, h): (usize, usize),
    taps: &Taps,
    tmp: &mut Tmp,
) -> bool {
    predict(Dst::Mid(dst), dst_stride, src, (w, h), taps, tmp)
}

/// One intermediate block back to samples: `Clip1((x + 32) >> 6)`.
#[inline]
pub fn uni(
    dst: &mut [u8],
    dst_stride: usize,
    a: &[i16],
    a_stride: usize,
    (w, h): (usize, usize),
) -> bool {
    let fits = matches!(w, 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64)
        && h >= 1
        && (h - 1) * dst_stride + w <= dst.len()
        && (h - 1) * a_stride + w <= a.len();
    if !fits || !accelerated() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `accelerated` checked at run time that the CPU has AVX2.
        #[allow(unsafe_code)]
        unsafe {
            avx2::uni(dst, dst_stride, a, a_stride, (w, h));
        }
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The average of two intermediate blocks: `Clip1((a + b + 64) >> 7)`.
#[inline]
pub fn bi(
    dst: &mut [u8],
    dst_stride: usize,
    (a, b): (&[i16], &[i16]),
    ab_stride: usize,
    (w, h): (usize, usize),
) -> bool {
    let fits = matches!(w, 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64)
        && h >= 1
        && (h - 1) * dst_stride + w <= dst.len()
        && (h - 1) * ab_stride + w <= a.len().min(b.len());
    if !fits || !accelerated() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `accelerated` checked at run time that the CPU has AVX2.
        #[allow(unsafe_code)]
        unsafe {
            avx2::bi(dst, dst_stride, (a, b), ab_stride, (w, h));
        }
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
