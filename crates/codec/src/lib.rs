//! Codec ids, packets, and stream parameters.
//!
//! This crate does not open a file and it does not decode. A container
//! turns a file into packets. A codec crate turns those packets into
//! pictures or samples.

#![forbid(unsafe_code)]

mod id;

pub use id::{CodecId, ContainerId, StreamCodec, TagError, UnknownCodec};
