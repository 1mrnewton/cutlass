//! A hint that the decoder will soon read a block of a reference picture:
//! inter prediction reads rows a stride apart, more of them at once than
//! the hardware prefetchers follow, and at 10 and 12 bits twice the bytes.
//! Asking for the next block's rows while the current one is predicted
//! overlaps their cache misses with its work.
//!
//! A prefetch reads no memory the program can see and cannot fault, so
//! what it hints changes nothing but timing; it is still given only
//! addresses inside the slice.

/// Hints rows `0..rows` of a region of `s`, row `r` being `s[at + r *
/// stride..][..w]`, into the cache. Rows past the end of `s` are left
/// out.
#[inline]
pub fn rect<T>(s: &[T], at: usize, stride: usize, w: usize, rows: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        let size = size_of::<T>().max(1);
        // One hint per 64-byte line: a row of `w` from anywhere in a line
        // spans at most this many.
        let lines = (w * size).div_ceil(64) + 1;
        let step = 64 / size;
        for r in 0..rows {
            let row = at + r * stride;
            for k in 0..lines {
                let i = row + k * step;
                if i >= s.len() {
                    return;
                }
                line(&s[i]);
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (s, at, stride, w, rows);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn line<T>(v: &T) {
    use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
    // SAFETY: SSE is part of x86-64; a prefetch of a live reference's
    // address reads nothing the program sees and cannot fault.
    #[allow(unsafe_code)]
    unsafe {
        _mm_prefetch::<_MM_HINT_T0>(core::ptr::from_ref(v).cast());
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn stays_inside_the_slice() {
        let s = vec![0u16; 1000];
        super::rect(&s, 990, 100, 64, 50);
        super::rect(&s, 0, 100, 64, 10);
        super::rect::<u16>(&[], 0, 1, 1, 1);
    }
}
