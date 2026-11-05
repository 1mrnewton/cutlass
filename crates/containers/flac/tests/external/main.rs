//! Tests against independent tools and real files.
//!
//! `clippy.toml` allows `expect` and `panic` in `#[cfg(test)]` modules.
//! This crate is the test harness, so the same exception is named here.
#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::print_stderr
)]

mod e2e;
mod fuzz;
mod oracle;
// Signals, WAV files, and tool runners shared with the codec's tests.
#[allow(dead_code)]
#[path = "../../../../codecs/flac/tests/external/support.rs"]
mod support;
