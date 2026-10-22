//! MediaCodec video and audio decode and encode for Android.
//!
//! The startup query lists the regular codecs. It does not decode or encode
//! a frame.

#![cfg(target_os = "android")]

mod caps;
mod jni;

pub use caps::Capabilities;
