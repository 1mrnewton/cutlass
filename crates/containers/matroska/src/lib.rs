//! Matroska and WebM framing.

#![forbid(unsafe_code)]

pub mod ebml;
mod error;
pub mod ids;
pub mod limits;
pub mod meta;

pub use error::Error;
