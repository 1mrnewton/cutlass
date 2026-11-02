//! Metal device, command queue, and retained RGBA textures.
//!
//! The parent crate compiles this module on macOS and iOS and allows
//! `unsafe_code` here. Submodules inherit that allow. `new*` objects are
//! released by [`Backend`]. Command buffers and blit encoders stay
//! autoreleased inside the pool around one submission.

mod backend;
mod sys;

pub use backend::{Backend, MetalError};
