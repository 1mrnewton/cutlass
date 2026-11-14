//! The 8-tap sub-sample interpolation of VP9 inter prediction for 8-bit
//! samples (libvpx `vpx_convolve8`): across, then down, each pass
//! rounded by 7 bits and held to 0..=255, and the average of two
//! predictions, `(a + b + 1) >> 1`.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(test)]
mod tests;

/// An 8-tap filter, with its taps paired the way the kernels multiply
/// them, worked out once (a `const fn`, for tables).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Filter {
    taps: [i16; 8],
    /// Taps `2k` and `2k + 1` as the signed byte pair `pmaddubsw` takes.
    pairs: [i16; 4],
    /// Whether every tap fits a signed byte (the kernels need it).
    bytes: bool,
}

impl Filter {
    pub const fn new(taps: [i16; 8]) -> Self {
        let mut pairs = [0i16; 4];
        let mut bytes = true;
        let mut k = 0;
        while k < 8 {
            if taps[k] < -128 || taps[k] > 127 {
                bytes = false;
            }
            k += 1;
        }
        let mut k = 0;
        while k < 4 {
            let lo = taps[2 * k] as i8 as u8 as u16;
            let hi = taps[2 * k + 1] as i8 as u8 as u16;
            pairs[k] = (lo | (hi << 8)) as i16;
            k += 1;
        }
        Self { taps, pairs, bytes }
    }

    pub const fn taps(&self) -> &[i16; 8] {
        &self.taps
    }
}

/// A block's prediction: `w`×`h` samples to `dst` (rows `dst_stride`
/// apart) from `src`, whose element `at` is the block's first reference
/// sample (rows `src_stride` apart), with the filter `fx` across and `fy`
/// down (`None`: no fraction). With `avg` the prediction is averaged into
/// what `dst` holds.
pub struct Block<'a> {
    pub dst: &'a mut [u8],
    pub dst_stride: usize,
    pub src: &'a [u8],
    pub at: usize,
    pub src_stride: usize,
    pub w: usize,
    pub h: usize,
    pub fx: Option<Filter>,
    pub fy: Option<Filter>,
    pub avg: bool,
    /// Scratch for the rows filtered across before the filter down:
    /// [`TMP`] samples.
    pub tmp: &'a mut [u8],
}

/// The scratch a [`Block`] needs.
pub const TMP: usize = 71 * 96;

// Its checks guard the x86-64 kernels.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
impl Block<'_> {
    /// Whether every load and store stays inside the buffers. The kernels
    /// read 16 samples from 3 before each 8 or 16 of a row, and 3 rows
    /// above and 4 below with a fraction down.
    #[inline(always)]
    fn fits(&self) -> bool {
        if !matches!(self.w, 4 | 8 | 16 | 32 | 64)
            || self.h == 0
            || self.h > 64
            || self.tmp.len() < TMP
        {
            return false;
        }
        if self.dst_stride < self.w || (self.h - 1) * self.dst_stride + self.w > self.dst.len() {
            return false;
        }
        let (above, below) = if self.fy.is_some() { (3, 4) } else { (0, 0) };
        let left = if self.fx.is_some() { 3 } else { 0 };
        let Some(first) = self
            .at
            .checked_sub(above * self.src_stride)
            .and_then(|v| v.checked_sub(left))
        else {
            return false;
        };
        let last_row = first + (self.h - 1 + above + below) * self.src_stride;
        last_row + self.w + 24 <= self.src.len()
            && self.fx.is_none_or(|f| f.bytes)
            && self.fy.is_none_or(|f| f.bytes)
    }
}

/// Predicts `b` with vector instructions. `false`, having written nothing,
/// when the CPU or the block is not one the kernels take.
#[must_use]
#[inline(always)]
// The `mut` serves the x86-64 kernels; rebinding it instead moved the
// whole struct through a `memcpy` call on every one.
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
pub fn predict(mut b: Block<'_>) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if super::avx2() && b.fits() {
            // SAFETY: `avx2` checked at run time that the CPU has AVX2,
            // the only feature `avx2::predict` enables.
            #[allow(unsafe_code)]
            unsafe {
                avx2::predict(&mut b);
            }
            return true;
        }
    }
    let _ = b;
    false
}

/// The scalar twin of [`predict`], from the specification.
pub fn predict_scalar(b: Block<'_>) {
    let Block {
        dst,
        dst_stride,
        src,
        at,
        src_stride,
        w,
        h,
        fx,
        fy,
        avg,
        tmp: _,
    } = b;
    let tap = |f: &[i16; 8], s: [u8; 8]| -> u8 {
        let mut sum = 0i32;
        for (k, &v) in s.iter().enumerate() {
            sum += i32::from(f[k]) * i32::from(v);
        }
        ((sum + 64) >> 7).clamp(0, 255) as u8
    };
    let rows = h + if fy.is_some() { 7 } else { 0 };
    let top = at - if fy.is_some() { 3 * src_stride } else { 0 };
    let mut tmp = vec![0u8; rows * w];
    for r in 0..rows {
        for c in 0..w {
            let p = top + r * src_stride + c;
            tmp[r * w + c] = match &fx {
                Some(f) => tap(&f.taps, core::array::from_fn(|k| src[p + k - 3])),
                None => src[p],
            };
        }
    }
    for r in 0..h {
        for c in 0..w {
            let v = match &fy {
                Some(f) => tap(&f.taps, core::array::from_fn(|k| tmp[(r + k) * w + c])),
                None => tmp[r * w + c],
            };
            let d = &mut dst[r * dst_stride + c];
            *d = if avg {
                ((u16::from(*d) + u16::from(v) + 1) >> 1) as u8
            } else {
                v
            };
        }
    }
}
