//! Integration tests: oracles, real files, the fuzz campaign and the seek
//! measurement.
//!
//! `clippy.toml` allows `expect` and `panic` in `#[cfg(test)]` modules.
//! This file is the harness, so the same exception is named here.
#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::unwrap_used
)]

mod local;
mod oracle;
mod support;
