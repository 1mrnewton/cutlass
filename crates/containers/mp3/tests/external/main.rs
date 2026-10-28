//! Integration tests.
//!
//! `clippy.toml` allows `expect` and `panic` in `#[cfg(test)]` modules.
//! This crate is the test harness, so the same exception is named here.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

mod fuzz;
mod local;
mod measure;
mod oracle;
mod support;
