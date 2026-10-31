//! One immutable snapshot at one time.
//!
//! `frame` returns the composited picture. It does not write the draft.
//! A video segment opens through the caller’s reader factory. [`OpenClips`]
//! keeps that file for one draft revision and seeks it at a later time.
//! A compound segment is the inlined child draft's picture, placed with the
//! compound clip. `audio_slice` is the engine mix at that time, and a
//! compound segment contributes the child's mix. Text, captions, and effects
//! are not drawn on this path.

#![forbid(unsafe_code)]

mod audio;
mod frame;
mod media;
mod nest;
mod sources;

pub use audio::{AudioError, MixWindow, audio_slice};
pub use cutlass_compositor::{Device, Frame};
pub use cutlass_engine::{ClipPcm, ClockKind, MixResult, ReaderFactory};
pub use cutlass_model::Micros;
pub use frame::{EvalError, Snapshot, frame, wait_export_frame};
pub use media::OpenClips;
