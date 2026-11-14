//! Kernels of the VP9 decoder and encoder (`cutlass-vp9`), on x86-64
//! with AVX2:
//!
//! - [`mc`]: the 8-tap sub-sample filters of inter prediction for 8-bit
//!   samples, and the average of two predictions.
//!
//! Each entry returns `false`, having written nothing, when it does not
//! run on this CPU or does not take its input; the codec then does the
//! work itself.

pub mod frame;
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

/// Runs `f` compiled for AVX2 (with BMI1, BMI2, LZCNT and POPCNT) when the
/// CPU has those, else as built: plain loops written over arrays and
/// slices become 256-bit vector code.
///
/// The `unsafe` is the one the crate allows for kernels: a safe function
/// marked `#[target_feature]`, entered behind the run-time check that
/// proves the features are present. The body is the caller's safe
/// closure, so it cannot read or write out of bounds. The closure and
/// what it calls must be `#[inline(always)]` to be compiled for AVX2;
/// anything left out of line still runs correctly, at baseline width.
#[inline(always)]
pub fn wide<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(target_arch = "x86_64")]
    {
        if wide_ok() {
            // SAFETY: `wide_ok` checked at run time that the CPU has every
            // feature `avx2_body` enables.
            #[allow(unsafe_code)]
            return unsafe { avx2_body(f) };
        }
    }
    f()
}

/// Whether [`wide`] runs its closure compiled for AVX2. Checked once.
#[must_use]
#[inline]
pub fn wide_ok() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static HAS: AtomicU8 = AtomicU8::new(0);
    match HAS.load(Ordering::Relaxed) {
        0 => {
            #[cfg(target_arch = "x86_64")]
            let has = std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("bmi1")
                && std::is_x86_feature_detected!("bmi2")
                && std::is_x86_feature_detected!("lzcnt")
                && std::is_x86_feature_detected!("popcnt");
            #[cfg(not(target_arch = "x86_64"))]
            let has = false;
            HAS.store(if has { 2 } else { 1 }, Ordering::Relaxed);
            has
        }
        v => v == 2,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
fn avx2_body<R>(f: impl FnOnce() -> R) -> R {
    f()
}
