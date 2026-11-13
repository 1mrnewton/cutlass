//! Running plain loops with the widest vector instructions the CPU has.
//!
//! A decoder's sample loops (filters over a row, sums of two predictions,
//! clamps) are written as ordinary safe Rust over slices, in a shape the
//! compiler turns into vector code. Built for the x86-64 baseline, that
//! code uses 128-bit SSE2 registers. [`wide`] runs such a loop compiled a
//! second time for AVX2, 256 bits, when the CPU has AVX2, and as built
//! otherwise.
//!
//! The `unsafe` here is the one the crate allows for kernels: a safe
//! function marked `#[target_feature]`, entered once behind the run-time
//! check that proves the feature is present. The function body is the
//! caller's closure, which is safe code: it indexes with bounds checks and
//! cannot read or write out of bounds. A mistake here could only run an
//! instruction the CPU lacks.
//!
//! The closure must be inlined into the AVX2 function to be compiled for
//! AVX2: callers mark it `#[inline(always)]` and keep the work inside it
//! in `#[inline(always)]` functions. Anything the compiler declines to
//! inline still runs correctly, at baseline width.

/// Runs `f`, compiled for AVX2 (with BMI1, BMI2, LZCNT and POPCNT) when
/// the CPU has those, else as built.
#[inline(always)]
pub fn wide<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: `has_avx2` checked at run time that the CPU has every
            // feature `avx2_body` enables.
            #[allow(unsafe_code)]
            return unsafe { avx2_body(f) };
        }
    }
    f()
}

/// Whether [`wide`] runs its closure compiled for AVX2.
#[must_use]
pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("bmi1")
            && std::is_x86_feature_detected!("bmi2")
            && std::is_x86_feature_detected!("lzcnt")
            && std::is_x86_feature_detected!("popcnt")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
fn avx2_body<R>(f: impl FnOnce() -> R) -> R {
    f()
}

#[cfg(test)]
mod tests {
    use super::wide;

    #[test]
    fn runs_the_closure_once_with_its_result() {
        let mut calls = 0;
        let v: Vec<i32> = (0..100).collect();
        let sum = wide(
            #[inline(always)]
            || {
                calls += 1;
                v.iter().map(|x| x * 3).sum::<i32>()
            },
        );
        assert_eq!((sum, calls), (14850, 1));
    }
}
