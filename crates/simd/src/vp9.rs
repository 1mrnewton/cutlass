//! Kernels of the VP9 decoder and encoder (`cutlass-vp9`), on x86-64
//! with AVX2:
//!
//! - [`mc`]: the 8-tap sub-sample filters of inter prediction for 8-bit
//!   samples, and the average of two predictions.
//!
//! Each entry returns `false`, having written nothing, when it does not
//! run on this CPU or does not take its input; the codec then does the
//! work itself.

pub mod lf;
pub mod mc;

/// Whether the CPU has AVX2. Checked once.
#[must_use]
#[inline]
pub fn avx2() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    /// 0: not checked yet, 1: no, 2: yes.
    static HAS: AtomicU8 = AtomicU8::new(0);
    match HAS.load(Ordering::Relaxed) {
        0 => {
            #[cfg(target_arch = "x86_64")]
            let has = std::is_x86_feature_detected!("avx2");
            #[cfg(not(target_arch = "x86_64"))]
            let has = false;
            HAS.store(if has { 2 } else { 1 }, Ordering::Relaxed);
            has
        }
        v => v == 2,
    }
}
