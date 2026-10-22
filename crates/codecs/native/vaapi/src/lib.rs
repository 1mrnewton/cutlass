//! VA-API video decode and encode for Intel and AMD GPUs on Linux.
//!
//! The startup query lists the codecs a render node claims. It does not
//! decode or encode a frame.

#![cfg(target_os = "linux")]

mod caps;

pub use caps::Capabilities;
