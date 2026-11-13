//! Fractional sample interpolation of HEVC pictures of 9 to 12 bits
//! (clause 8.5.3.3.3 of ITU-T H.265) into the 14-bit intermediates of the
//! weighted sample prediction, as [`super::mc::put`] does for 8 bits.
//!
//! The entry returns `false`, having written nothing, for a block it does
//! not take (its width, its bit depth, or a source too short for the
//! loads): the caller then predicts the block its own way. The test
//! compares it with a scalar version written from the clause.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(test)]
mod tests;

pub use super::mc::Taps;

/// The reference samples of one block, as [`super::mc::Src`] for 8 bits:
/// `src[at]` is the first sample of the window the filters read, `T / 2 -
/// 1` rows above and columns left of the block's first sample.
#[derive(Clone, Copy, Debug)]
pub struct Src<'a> {
    pub src: &'a [u16],
    pub at: usize,
    pub stride: usize,
}

impl Src<'_> {
    /// Where row `j` of the window starts.
    #[inline]
    #[must_use]
    pub fn row(&self, j: usize) -> usize {
        self.at + j * self.stride
    }

    /// Whether every load of a `w`×`h` block stays inside `src`: the
    /// kernel reads `w + 24` samples of each of `h + taps - 1` rows.
    #[inline]
    fn holds(&self, w: usize, h: usize, taps: usize) -> bool {
        (h + taps - 2)
            .checked_mul(self.stride)
            .and_then(|r| r.checked_add(self.at))
            .and_then(|r| r.checked_add(w + 24))
            .is_some_and(|end| end <= self.src.len())
    }
}

/// Scratch rows the kernel keeps between blocks, made on first use.
#[derive(Default)]
pub struct Tmp {
    #[cfg(target_arch = "x86_64")]
    rows: Vec<avx2::Strips>,
}

/// Where [`predict`] puts a block, rows `dst_stride` apart.
pub enum Dst<'a> {
    /// The 14-bit intermediates ([`put`]).
    Mid(&'a mut [i16]),
    /// Samples, the default weighting of this list alone.
    Uni(&'a mut [u16]),
    /// Samples, the default weighting of this list with the other list's
    /// intermediates, rows `w` apart.
    Bi(&'a mut [u16], &'a [i16]),
}

/// The intermediates of a `w`×`h` block of samples of `bit_depth` bits
/// (9 to 12) into `dst` (rows `dst_stride` apart): `x << (14 - bit_depth)`
/// with no fraction, the sums of the taps shifted right by `bit_depth - 8`
/// with one, and with both the vertical sums of those shifted right by 6.
#[inline]
pub fn put(
    dst: &mut [i16],
    dst_stride: usize,
    src: &Src<'_>,
    (w, h): (usize, usize),
    taps: &Taps,
    bit_depth: u32,
    tmp: &mut Tmp,
) -> bool {
    predict(Dst::Mid(dst), dst_stride, src, (w, h), taps, bit_depth, tmp)
}

/// The intermediates of [`put`], or the samples they make as [`store`]
/// makes them, in one pass.
#[inline]
pub fn predict(
    dst: Dst<'_>,
    dst_stride: usize,
    src: &Src<'_>,
    (w, h): (usize, usize),
    taps: &Taps,
    bit_depth: u32,
    tmp: &mut Tmp,
) -> bool {
    let room = |len: usize| (h - 1) * dst_stride + w <= len;
    let fits = matches!(w, 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64)
        && (1..=64).contains(&h)
        && (9..=12).contains(&bit_depth)
        && matches!(taps.len, 4 | 8)
        && dst_stride >= w
        && match &dst {
            Dst::Mid(d) => room(d.len()),
            Dst::Uni(d) => room(d.len()),
            Dst::Bi(d, o) => room(d.len()) && o.len() >= w * h,
        }
        && src.holds(w, h, taps.len);
    if !fits || !super::mc::accelerated() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `accelerated` checked at run time that the CPU has AVX2.
        #[allow(unsafe_code)]
        unsafe {
            let (mut out, mode) = avx2::Sink::new(dst, dst_stride, w, bit_depth);
            let args = (src, (w, h), taps, bit_depth, &mut tmp.rows);
            match mode {
                0 => avx2::predict::<0>(&mut out, args.0, args.1, args.2, args.3, args.4),
                1 => avx2::predict::<1>(&mut out, args.0, args.1, args.2, args.3, args.4),
                _ => avx2::predict::<2>(&mut out, args.0, args.1, args.2, args.3, args.4),
            }
        }
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (tmp, dst);
        false
    }
}

/// The default weighted prediction (clause 8.5.3.3.4.2) of a `w`×`h`
/// block of `bit_depth` bits (9 to 12) into `dst` (rows `dst_stride`
/// apart): from one block of intermediates `(a + o1) >> (14 - bit_depth)`,
/// from two `(a + b + o2) >> (15 - bit_depth)`, clipped to the sample
/// range. The intermediates' rows lie `ab_stride` apart.
#[inline]
pub fn store(
    dst: &mut [u16],
    dst_stride: usize,
    (a, b): (&[i16], Option<&[i16]>),
    ab_stride: usize,
    (w, h): (usize, usize),
    bit_depth: u32,
) -> bool {
    let fits = w % 4 == 0
        && (4..=64).contains(&w)
        && h >= 1
        && (9..=12).contains(&bit_depth)
        && dst_stride >= w
        && ab_stride >= w
        && (h - 1) * dst_stride + w <= dst.len()
        && (h - 1) * ab_stride + w <= a.len().min(b.map_or(usize::MAX, <[i16]>::len));
    if !fits || !super::mc::accelerated() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `accelerated` checked at run time that the CPU has AVX2.
        #[allow(unsafe_code)]
        unsafe {
            avx2::store(dst, dst_stride, (a, b), ab_stride, (w, h), bit_depth);
        }
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
