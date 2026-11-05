//! VP8 decoder and encoder (RFC 6386).
//!
//! The decoder accepts every VP8 bitstream version. The encoder emits
//! version 0. A WebM container is outside this crate.

#![forbid(unsafe_code)]

mod bool_coder;
mod header;
mod tables;
