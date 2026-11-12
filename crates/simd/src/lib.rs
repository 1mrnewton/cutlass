//! Vector kernels for the Rust codecs: the sample loops a decoder spends
//! its time in, written with the CPU's vector instructions.
//!
//! This is the one crate outside the platform crates that may write
//! `unsafe`, and for kernels only in one shape. A kernel is a safe
//! function marked `#[target_feature]`. It takes fixed-size arrays and
//! returns values: it has no pointer, no slice index computed from the
//! stream, and no load or store intrinsic. The compiler still requires
//! `unsafe` to call such a function from ordinary code, because the caller
//! must promise the CPU has the feature. Each kernel's entry makes that
//! one call, behind a `cfg` or a run-time check that proves the feature is
//! present. A mistake there can run an instruction the CPU lacks. It
//! cannot read or write out of bounds.
//!
//! Every kernel has a scalar twin written from the specification, and a
//! test that compares the two on random and edge inputs. A target without
//! a vector kernel runs the twin.
//!
//! The one other `unsafe` is [`shared_rows`]: a picture's final rows that
//! one writer publishes top to bottom while other threads read the rows
//! published so far, without copying them per reader.

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;

pub mod av1;
pub mod shared_rows;

/// Whether this build has vector kernels. Without them every entry runs
/// its scalar twin, and a caller with its own fast path may prefer that.
pub const ACCELERATED: bool = cfg!(all(target_arch = "aarch64", target_feature = "neon"));
