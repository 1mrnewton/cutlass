//! The media graph under the draft.
//!
//! The shell registers container and decoder implementations. This crate
//! opens them, maps timeline time, caches pictures, and mixes audio.
//! Filmstrips, waveforms, preload, and nested playback are not in this crate yet.

#![forbid(unsafe_code)]

mod cache;
mod clock;
mod error;
mod map;
mod mix;
mod reader;
mod source;

pub use cache::{FrameWindow, MAX_WINDOW};
pub use clock::{ClockKind, MediaClock};
pub use error::EngineError;
pub use map::{
    MapError, MapSpeed, TimelineMap, map_timeline, media_path, micros_to_ticks, placement_of,
    speed_at, ticks_to_micros,
};
pub use mix::{
    AudioSample, ClipPcm, ClipRole, Mix, MixClip, MixResult, mix_inputs, render, volume_at,
};
pub use reader::{
    DecodedFrame, HeldFrame, Media, MediaInfo, PcmReady, ReaderFactory, StreamReport,
};
pub use source::{VideoSource, open_segment};

#[cfg(test)]
mod tests;
