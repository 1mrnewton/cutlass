//! Cutlass editing data model: project, media pool, timeline, tracks, clips.
//!
//! Design goals:
//! - **Correct types**: strongly-typed IDs (no mixing a [`TrackId`] with a
//!   [`ClipId`]), OTIO-style [`RationalTime`] (exact NTSC rates) sourced from
//!   the shared [`cutlass_core`] contract, and explicit timeline-vs-source
//!   ranges each carrying their own rate.
//! - **Fast lookup**: entities are stored in hash maps keyed by ID, so finding
//!   a clip/track/media by ID is O(1). The timeline keeps a `ClipId -> TrackId`
//!   index so a clip can be located across all tracks without scanning.
//! - **Independent testing**: this crate has no dependency on decode/render, so
//!   the model can be exercised in isolation.
//!
//! A [`Project`] owns one [`Timeline`] and a media pool of [`MediaSource`]s.
//!
//! ## Templates
//!
//! A [`Template`] is a finished [`Project`] whose author has marked specific
//! clips as user-replaceable ([`Replaceable`]), mirroring CapCut: the slot
//! keeps its sample media (so the template previews), and applying the template
//! swaps the media while the slot's duration, transform, effects, transitions,
//! and animations stay locked.

mod capabilities;
mod clip;
mod effects;
mod error;
mod ids;
mod look;
mod media;
mod metadata;
mod param;
mod persist;
mod project;
mod schema;
mod serde_map;
mod sticker;
mod template;
pub mod template_bundle;
mod time;
mod timeline;
mod track;
mod transition;

/// Fast hash map for integer-keyed entities (FxHash; no DoS resistance needed
/// for an in-process editing model, but much faster than SipHash for `u64`).
pub type Map<K, V> = rustc_hash::FxHashMap<K, V>;

pub use capabilities::{ClipAction, ClipCapabilities};
pub use clip::{
    AnimatedTransform, Clip, ClipParam, ClipSource, ClipTransform, CropRect, Generator, LookParam,
    MAX_CLIP_VOLUME, MAX_SHAPE_DIM, MAX_SPEED, MAX_STAR_POINTS, MAX_STROKE_WIDTH,
    MIN_CROP_FRACTION, MIN_SPEED, ParamValue, Replaceable, SHAPE_DROP_HEIGHT, SHAPE_DROP_WIDTH,
    SPEED_CURVE_SCALE, Shape, ShapeParam, ShapePath, ShapePathPoint, ShapeStroke, SlotMedia,
    SpeedPresetSpec, TextAlignH, TextAlignV, TextBackground, TextCase, TextParam, TextShadow,
    TextStroke, TextStyle, audio_gain_at, look_animation_combo_period_ticks,
    look_animation_window_ticks, speed_curve_integral, speed_curve_source_fraction, speed_preset,
    speed_preset_catalog, speed_preset_id, validate_speed_curve, validate_volume,
    validate_volume_envelope,
};
pub use effects::{EffectInstance, EffectParamSpec, EffectSpec, effect_catalog, effect_spec};
pub use error::ModelError;
pub use ids::{ClipId, LinkId, MarkerId, MediaId, ProjectId, TemplateId, TrackId};
pub use look::{
    ANIMATION_INTENSITY_RANGE, ANIMATION_PARAM_DEFAULT, ANIMATION_SPEED_RANGE,
    ANIMATION_STAGGER_RANGE, AnimationKnobs, AnimationRef, AnimationSlot, AnimationSpec, AudioRole,
    ChromaKey, ColorAdjustments, Filter, FilterSpec, Lut, Mask, MaskKind, MaskSpec, StabilizeLevel,
    TextEffectSpec, animation_catalog, animation_spec, filter_catalog, filter_spec, mask_catalog,
    text_effect_catalog, text_effect_spec,
};
pub use media::{MediaKind, MediaSource, STILL_DEFAULT_DURATION_TICKS, STILL_TICK_RATE};
pub use metadata::ProjectMetadata;
pub use param::{EASING_PRESETS, Easing, EasingPreset, Keyframe, Lerp, Param, easing_preset};
pub use persist::{PROJECT_FILE_EXTENSION, PROJECT_FILE_VERSION};
pub use project::Project;
pub use schema::{PROJECT_SCHEMA_KIND, PROJECT_SCHEMA_VERSION, ProjectSchema};
pub use sticker::{StickerSpec, sticker_catalog, sticker_spec};
pub use template::{
    Pick, TEMPLATE_FILE_EXTENSION, TEMPLATE_SCHEMA_KIND, Template, TemplateCategory, TemplateMeta,
};
pub use time::{
    Rational, RationalTime, TimeError, TimeRange, check_same_rate, rate_eq, resample, time_add,
    time_sub,
};
pub use timeline::{CanvasAspect, CanvasSettings, Marker, MarkerColor, Timeline};
pub use track::{Track, TrackKind};
pub use transition::{
    DEFAULT_TRANSITION_TICKS, Transition, TransitionSpec, transition_catalog, transition_spec,
};
