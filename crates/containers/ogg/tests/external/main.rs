//! Integration tests against independent producers and demuxers.
//!
//! `clippy.toml` allows `expect` and `panic` in `#[cfg(test)]` modules.
//! This file is the harness, so the same exception is named here.
#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::unwrap_used
)]

mod measure;
mod mux_oracle;
mod oracle;
mod support;
mod vorbis_oracle;
