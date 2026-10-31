//! One immutable snapshot at one time.
//!
//! `frame` returns the composited picture. It does not write the draft.
//! A video segment opens through the caller’s reader factory. [`OpenClips`]
//! keeps that file for one draft revision and seeks it at a later time.
//! `audio_slice` is the engine mix at that time. Nested playback, text,
//! captions, and effects are not drawn on this path.

#![forbid(unsafe_code)]

mod audio;
mod frame;
mod media;
mod sources;

pub use audio::{AudioError, MixWindow, audio_slice};
pub use cutlass_compositor::{Device, Frame};
pub use cutlass_engine::{ClipPcm, ClockKind, MixResult, ReaderFactory};
pub use cutlass_model::Micros;
pub use frame::{EvalError, Snapshot, frame, wait_export_frame};
pub use media::OpenClips;
