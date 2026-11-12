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

/// Calls a kernel's vector version where the build has one, and its plain
/// version elsewhere. This is the crate's only `unsafe`.
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
        #[cfg(not(target_arch = "aarch64"))]
        {
            crate::scalar::$name($($argument),*)
        }
    }};
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

pub mod scalar;

#[cfg(test)]
mod tests {
    #[test]
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
