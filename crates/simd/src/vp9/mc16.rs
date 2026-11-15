//! The 8-tap sub-sample interpolation of VP9 inter prediction for 10- and
//! 12-bit samples (libvpx `vpx_highbd_convolve8`): across, then down, each
//! pass rounded by 7 bits and held to `0..=max`, and the average of two
//! predictions, `(a + b + 1) >> 1`.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(test)]
mod tests;

pub use super::mc::Filter;

/// A block's prediction, as [`super::mc::Block`] but on `u16` samples
/// held to `0..=max`. It needs no scratch.
pub struct Block<'a> {
    pub dst: &'a mut [u16],
    pub dst_stride: usize,
    pub src: &'a [u16],
    pub at: usize,
    pub src_stride: usize,
    pub w: usize,
    pub h: usize,
    pub fx: Option<&'a Filter>,
    pub fy: Option<&'a Filter>,
    pub avg: bool,
    /// The largest sample, `(1 << bit_depth) - 1`: at most 4095.
    pub max: u16,
}

// Its checks guard the x86-64 kernels.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
impl Block<'_> {
    /// Whether every load and store stays inside the buffers. The kernels
    /// read up to 24 samples from 3 before each 16 outputs of a row, and 3
    /// rows above and 4 below with a fraction down.
    #[inline(always)]
    fn fits(&self) -> bool {
        if !matches!(self.w, 4 | 8 | 16 | 32 | 64) || self.h == 0 || self.h > 64 || self.max > 4095
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
    }
}

/// Predicts `b` with vector instructions. `false`, having written nothing,
/// when the CPU or the block is not one the kernels take. Exact for
/// samples below 32768 (each is multiplied as a signed 16-bit value).
#[must_use]
#[inline(always)]
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
        max,
    } = b;
    let tap = |f: &[i16; 8], s: [u16; 8]| -> u16 {
        let mut sum = 0i32;
        for (k, &v) in s.iter().enumerate() {
            sum += i32::from(f[k]) * i32::from(v);
        }
        ((sum + 64) >> 7).clamp(0, i32::from(max)) as u16
    };
    let rows = h + if fy.is_some() { 7 } else { 0 };
    let top = at - if fy.is_some() { 3 * src_stride } else { 0 };
    let mut tmp = vec![0u16; rows * w];
    for r in 0..rows {
        for c in 0..w {
            let p = top + r * src_stride + c;
            tmp[r * w + c] = match &fx {
                Some(f) => tap(f.taps(), core::array::from_fn(|k| src[p + k - 3])),
                None => src[p],
            };
        }
    }
    for r in 0..h {
        for c in 0..w {
            let v = match &fy {
                Some(f) => tap(f.taps(), core::array::from_fn(|k| tmp[(r + k) * w + c])),
                None => tmp[r * w + c],
            };
            let d = &mut dst[r * dst_stride + c];
            *d = if avg {
                ((u32::from(*d) + u32::from(v) + 1) >> 1) as u16
            } else {
                v
            };
        }
    }
}
