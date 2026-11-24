//! The H.264 codec's sample loops, each written twice: a plain version in
//! [`scalar`], and a vector version for the CPU the build targets.
//!
//! # Memory safety
//!
//! No kernel takes a pointer or a length. Each takes fixed-size arrays and
//! reads and writes only those: a vector is loaded from an array and
//! stored back to one, through a load or store no wider than the array, so
//! its type keeps it in bounds. A wrong index in a kernel is a panic, not
//! a stray read or write.
//!
//! The whole-block NEON predictors in `neon_pel` and `neon_qpel`, and the
//! NEON filters of a macroblock's boundary alone in `neon`, are the
//! exception: they load and store through pointers into the caller's
//! slices, after one check per block that every byte they touch is inside
//! them. Their modules say what that check covers.
//!
//! [`growing`] is the other exception, and the one piece here that is not
//! a sample loop: a buffer one thread fills from the front while others
//! read what is filled, which needs `unsafe` to share. Its module says why
//! that is sound.
//!
//! The one `unsafe` per kernel is the call into its vector version. The
//! compiler asks for it because that function is built for a CPU feature,
//! and the caller has to promise the CPU has it. On 64-bit ARM the feature
//! is NEON, which every such CPU has. That promise is all the `unsafe`
//! covers; it says nothing about memory.
//!
//! Every vector kernel is tested against its [`scalar`] twin on random and
//! edge-like samples, and must give the same samples.

/// Calls a kernel's vector version where the build has one for the CPU it
/// runs on, and its plain version elsewhere. This is the crate's only
/// `unsafe` outside the vector modules.
macro_rules! kernel {
    ($name:ident($($argument:expr),*)) => {{
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: the callee is built for NEON, which every aarch64 CPU
            // has. It is otherwise a safe function.
            #[allow(unsafe_code)]
            unsafe {
                crate::neon::$name($($argument),*)
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            match crate::x86::level() {
                // SAFETY: the callee is built for AVX2 and the features
                // `level` found with it on this CPU. It is otherwise a
                // safe function.
                #[allow(unsafe_code)]
                crate::x86::Level::Avx2 => unsafe { crate::x86::avx2::$name($($argument),*) },
                // SAFETY: the callee is built for SSE4.1 and the features
                // `level` found with it on this CPU. It is otherwise a
                // safe function.
                #[allow(unsafe_code)]
                crate::x86::Level::Sse41 => unsafe { crate::x86::sse41::$name($($argument),*) },
                crate::x86::Level::Scalar => crate::scalar::$name($($argument),*),
            }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            crate::scalar::$name($($argument),*)
        }
    }};
}

/// Runs a test's `body` once on each vector version of the kernels this
/// CPU can run: on x86-64 the SSE4.1 one and, where the CPU has it, the
/// AVX2 one. Elsewhere the build has one version and `body` runs once.
#[cfg(test)]
pub(crate) fn each_tier(body: fn()) {
    #[cfg(target_arch = "x86_64")]
    x86::each_tier(body);
    #[cfg(not(target_arch = "x86_64"))]
    body();
}

/// Makes a test of each function named, which runs it through
/// [`each_tier`]. Used in a module inside the test module that holds the
/// functions.
#[cfg(test)]
macro_rules! tests_on_each_tier {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                crate::each_tier(super::$name);
            }
        )*
    };
}

/// Asks for `items[at]` to be brought into the cache before it is read,
/// the way [`mc::prefetch`] asks for a reference's rows. Nothing is asked
/// for when `at` is outside `items`, and nothing changes that a caller
/// could see.
#[inline]
pub fn prefetch_item<T>(items: &[T], at: usize) {
    kernel!(prefetch_item(items, at));
}

pub mod deblock;
mod edge;
pub mod growing;
pub mod intra;
pub mod mc;
mod mc_block;
pub mod residual;

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "aarch64")]
mod neon_intra;
#[cfg(target_arch = "aarch64")]
mod neon_mc;
#[cfg(target_arch = "aarch64")]
mod neon_pel;
#[cfg(target_arch = "aarch64")]
mod neon_qpel;
#[cfg(target_arch = "aarch64")]
mod neon_residual;

#[cfg(target_arch = "x86_64")]
mod x86;

pub mod scalar;

#[cfg(test)]
mod tests {
    mod on_each_tier {
        tests_on_each_tier!(asking_for_an_item_changes_nothing);
    }

    fn asking_for_an_item_changes_nothing() {
        // Items over several cache lines, a line exactly, and none at all;
        // places inside, at the end, and far past it.
        let wide: Vec<[u8; 340]> = (0..4u8).map(|seed| [seed; 340]).collect();
        let line = [[7u8; 64]; 3];
        let empty = [(); 5];
        for at in [0, 1, 3, 4, usize::MAX] {
            super::prefetch_item(&wide, at);
            super::prefetch_item(&line, at);
            super::prefetch_item(&empty, at);
            super::prefetch_item::<u32>(&[], at);
        }
        assert!(
            wide.iter()
                .zip(0u8..)
                .all(|(item, seed)| item.iter().all(|&byte| byte == seed))
        );
        assert!(line.iter().flatten().all(|&byte| byte == 7));
    }
}
