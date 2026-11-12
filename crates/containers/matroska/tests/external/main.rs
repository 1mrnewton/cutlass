//! Integration tests against independent tools, fuzz campaigns, local
//! media, and seek cost.
//!
//! `clippy.toml` allows `expect` and `panic` in `#[cfg(test)]` modules.
//! This file is the harness, so the same exception is named here.
#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::unwrap_used
)]

#[path = "../behavior/build.rs"]
#[allow(dead_code)]
mod build;
mod fuzz;
mod local;
mod measure;
mod oracle;
mod support;
