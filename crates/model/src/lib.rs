//! The draft: tracks, segments, materials, and microseconds.
//!
//! A draft is one document. Time is a signed count of microseconds.
//! Frame rate is a numerator and a denominator. Tracks hold segments.
//! A segment does not own its media: `material_id` and `extra_material_refs`
//! point into [`Materials`].
//!
//! On a video or mixed track, flag 0 is the main track. Any other flag is a
//! sub-video track. Track attribute is a bit field: mute, hide, lock, solo,
//! and muted by solo.

#![forbid(unsafe_code)]

mod clip;
mod draft;
mod enums;
mod id;
mod keyframe;
mod materials;
mod meta;
mod segment;
mod serde_util;
mod time;
mod track;

pub use clip::{Clip, Flip, Point, UniformScale};
pub use draft::{CanvasConfig, Draft, DraftConfig};
pub use enums::{
    AnimType, ApplyType, CanvasRatio, CropRatio, HdrDisplayMode, HdrType, KeyframeCurve,
    RenderIndexMode, SegmentSource, SpeedMode, TrackAttribute, TrackFlag, TrackType,
};
pub use id::{DraftId, MaterialId, SegmentId, TrackId};
pub use keyframe::{CommonKeyframe, CommonKeyframeGroup};
pub use materials::{
    ConstantSpeedError, CurveSpeed, MaterialAnimations, MaterialAudio, MaterialAudioFade,
    MaterialCanvas, MaterialColorConfig, MaterialDraft, MaterialEffect, MaterialHsl,
    MaterialPictureAdjust, MaterialPluginEffect, MaterialRef, MaterialShape, MaterialSpeed,
    MaterialText, MaterialTransition, MaterialVideo, MaterialVideoEffect, Materials, RawMaterial,
    SpeedPoint, StickerAnimation, WordColumns, constant_target_duration, curve_speed_factor,
    curve_target_duration,
};
pub use meta::MetaType;
pub use segment::{HdrSettings, ResponsiveLayout, Segment};
pub use time::{FrameError, FrameRate, Micros, RangeError, TimeRange, TimecodeLabel};
pub use track::Track;
