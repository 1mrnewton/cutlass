//! The H.264 codec's sample loops, each written twice: a plain version in
//! [`scalar`], and a vector version for the CPU the build targets.
//!
//! # Memory safety
//!
//! No kernel takes a pointer or a length. Each takes fixed-size arrays and
//! reads and writes only those, in safe Rust: a vector is built from an
//! array's bytes and written back to one, never loaded through an address.
//! A wrong index in a kernel is a panic, not a stray read or write.
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

pub mod deblock;
mod edge;
pub mod mc;
mod mc_block;

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "aarch64")]
mod neon_mc;

pub mod scalar;
