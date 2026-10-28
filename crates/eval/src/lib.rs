//! One immutable snapshot at one time.
//!
//! `frame` returns the composited picture. It does not write the draft.
//! `engine`, `text`, `captions`, and `effects` stay on the dependency list
//! and are not called. Nested playback, text, captions, and effects are not
//! part of this path.

#![forbid(unsafe_code)]

mod frame;
mod sources;

pub use frame::{EvalError, Snapshot, frame};
