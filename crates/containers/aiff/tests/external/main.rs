//! Files from ffmpeg, read back by this crate, and this crate's files read by ffmpeg.
//!
//! ffprobe is the same demuxer as ffmpeg. `afinfo` is a separate reader when
//! it is on PATH.

#![allow(clippy::expect_used, clippy::panic, clippy::print_stderr)]

mod local;
mod oracle;
