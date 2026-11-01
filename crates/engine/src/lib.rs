//! The media graph under the draft.
//!
//! The shell registers container and decoder implementations. This crate
//! opens them, maps timeline time, caches pictures, and mixes audio.
//! A filmstrip is a smaller picture cache. Asking before a fill is not ready
//! and does not decode. Waveforms are not in this crate yet. Nested playback
//! is not in this crate.

#![forbid(unsafe_code)]

mod cache;
mod clock;
mod error;
mod filmstrip;
mod map;
mod mix;
mod preload;
mod reader;
mod source;

pub use cache::{FrameWindow, MAX_WINDOW};
pub use clock::{ClockKind, MediaClock};
pub use cutlass_container::DisplayQuarter;
pub use error::EngineError;
pub use filmstrip::{MAX_EDGE, MAX_STRIP, StripCell};
pub use map::{
    MapError, MapSpeed, TimelineMap, is_graphic, map_timeline, media_path, micros_to_ticks,
    placement_of, speed_at, ticks_to_micros,
};
pub use mix::{
    AudioSample, ClipPcm, ClipRole, Mix, MixClip, MixResult, mix_inputs, render, volume_at,
};
pub use reader::{
    DecodedFrame, HeldFrame, Media, MediaInfo, PcmReady, ReaderFactory, StreamReport,
};
pub use source::{VideoSource, decoded_picture, open_segment};

#[cfg(test)]
mod tests;
