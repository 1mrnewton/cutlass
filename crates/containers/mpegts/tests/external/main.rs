//! Integration tests against independent producers and demuxers: FFmpeg
//! 8.1 and GStreamer 1.28 write the inputs, and ffprobe, ffmpeg and
//! GStreamer's `tsdemux` read this crate's output. Each test skips, with a
//! message, when a tool it needs is missing.
//!
//! `clippy.toml` allows `expect` and `panic` in `#[cfg(test)]` modules.
//! This file is the harness, so the same exception is named here.
#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::unwrap_used
)]

mod demux_oracle;
mod local;
mod mux_oracle;
mod support;
