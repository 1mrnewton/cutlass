//! The 8-tap sub-sample interpolation of VP9 inter prediction for 8-bit
//! samples (libvpx `vpx_convolve8`): across, then down, each pass
//! rounded by 7 bits and held to 0..=255, and the average of two
//! predictions, `(a + b + 1) >> 1`.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(test)]
mod tests;

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
    pub fx: Option<[i16; 8]>,
    pub fy: Option<[i16; 8]>,
    pub avg: bool,
    /// Scratch for the rows filtered across before the filter down:
    /// [`TMP`] samples.
    pub tmp: &'a mut [u8],
}

/// The scratch a [`Block`] needs.
pub const TMP: usize = 71 * 96;

impl Block<'_> {
    /// Whether every load and store stays inside the buffers. The kernels
    /// read 16 samples from 3 before each 8 or 16 of a row, and 3 rows
    /// above and 4 below with a fraction down.
    fn fits(&self) -> bool {
        if !matches!(self.w, 4 | 8 | 16 | 32 | 64) || self.h == 0 || self.h > 64 || self.tmp.len() < TMP {
            return false;
        }
        if self.dst_stride < self.w
            || (self.h - 1) * self.dst_stride + self.w > self.dst.len()
        {
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
            && self.fx.is_none_or(ok_taps)
            && self.fy.is_none_or(ok_taps)
    }
}

/// Taps the kernels take: each fits a signed byte.
fn ok_taps(f: [i16; 8]) -> bool {
    f.iter().all(|&t| (-128..=127).contains(&t))
}

/// Predicts `b` with vector instructions. `false`, having written nothing,
/// when the CPU or the block is not one the kernels take.
#[must_use]
pub fn predict(b: Block<'_>) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if super::avx2() && b.fits() {
            // SAFETY: `avx2` checked at run time that the CPU has AVX2,
            // the only feature `avx2::predict` enables.
            #[allow(unsafe_code)]
            unsafe {
                avx2::predict(b);
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
                Some(f) => tap(f, core::array::from_fn(|k| src[p + k - 3])),
                None => src[p],
            };
        }
    }
    for r in 0..h {
        for c in 0..w {
            let v = match &fy {
                Some(f) => tap(f, core::array::from_fn(|k| tmp[(r + k) * w + c])),
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
