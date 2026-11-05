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

mod decode_oracle;
mod encode_oracle;
mod fuzz;
mod support;
