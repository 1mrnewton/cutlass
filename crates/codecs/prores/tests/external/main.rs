//! Tests against independent tools and real files.
//!
//! ffmpeg makes the streams and decodes them for reference. The clips come
//! from `../local-files`; a test returns early when the folder or a tool is
//! missing, unless `PRORES_REQUIRE_ORACLES` is set.
//!
//! `clippy.toml` allows `expect` and `panic` in `#[cfg(test)]` modules.
//! This crate is the test harness, so the same exception is named here.
#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::print_stderr
)]

mod decode_oracle;
mod support;
