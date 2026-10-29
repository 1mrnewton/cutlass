//! One immutable snapshot at one time.
//!
//! `frame` returns the composited picture. It does not write the draft.
//! `audio_slice` is the engine mix at that time. Nested playback, text,
//! captions, and effects are not drawn on this path.

#![forbid(unsafe_code)]

mod audio;
mod frame;
mod sources;

pub use audio::{AudioError, MixWindow, audio_slice};
pub use cutlass_compositor::{Device, Frame};
pub use cutlass_engine::{ClipPcm, ClockKind, MixResult};
pub use cutlass_model::Micros;
pub use frame::{EvalError, Snapshot, frame, wait_export_frame};
