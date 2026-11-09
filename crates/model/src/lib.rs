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

pub mod animation;
mod bin;
mod clip;
mod draft;
mod enums;
pub mod graphs;
mod id;
mod keyframe;
mod linkage;
mod materials;
mod meta;
mod relationship;
mod segment;
mod serde_util;
mod shape_keys;
mod text_keys;
mod time;
mod track;
mod tracking;

pub use animation::{
    ANIMATION_PRESETS, AnimationPreset, Letter, LetterMotion, LetterPose, LetterStep, LoopMotion,
    Motion, animated_at, animation_preset, letters_at, segment_animations,
};
pub use bin::{BinFolder, BinLink, BinPath, VirtualStore};
pub use clip::{Clip, Flip, Point, UniformScale};
pub use draft::{CanvasConfig, Draft, DraftConfig};
pub use enums::{
    AnimType, ApplyType, CanvasRatio, CropRatio, HdrDisplayMode, HdrType, KeyframeCurve,
    RenderIndexMode, SegmentSource, SpeedMode, TrackAttribute, TrackFlag, TrackType,
};
pub use graphs::{GRAPH_PRESETS, GraphPreset};
pub use id::{DraftId, MaterialId, SegmentId, TrackId};
pub use keyframe::{
    ADJUST_KEYS, ALPHA_KEY, CANVAS_COLOR_KEYS, CHROMA_SHADOW_KEYS, CHROMA_STRENGTH_KEYS,
    CommonKeyframe, CommonKeyframeGroup, EFFECT_CONTROL_PREFIX, FILTER_STRENGTH_KEYS, GraphPoint,
    HSL_KEYS, KeyframeGraph, LUT_STRENGTH_KEYS, MASK_FEATHER_KEYS, MASK_KEYS, MASK_POSITION_KEYS,
    MASK_ROTATION_KEYS, MASK_ROUND_KEYS, MASK_SIZE_KEYS, OPACITY_KEYS, POSE_KEYS, POSITION_KEYS,
    RELIGHT_ANGLE_KEYS, RELIGHT_COLOR_KEYS, RELIGHT_STRENGTH_KEYS, ROTATION_KEYS, SCALE_KEYS,
    TEXT_COLOR_KEYS, VOLUME_KEY, WHEEL_KEYS, add_keys, clip_at, effect_control_id,
    effect_control_key, has_keys, hex_color_at, is_keyable, is_pose_key, is_unit_colour_key,
    key_offsets, key_offsets_where, put_key, relight_at, remove_keys, remove_keys_where, sample,
    shift_keys, text_color_at,
};
pub use linkage::{KindSwitches, LinkageSettings, linkage_from_biz};
pub use materials::{
    Algorithm, BloomParams, Bubble, CAPTION_STYLE_KEYS, CaptionTemplateInfo, ConstantSpeedError,
    Crop, CurveSpeed, FreezeInfo, Kept, MaterialAnimations, MaterialAudio, MaterialAudioFade,
    MaterialCanvas, MaterialColorConfig, MaterialDraft, MaterialEffect, MaterialHsl,
    MaterialPictureAdjust, MaterialPluginEffect, MaterialRef, MaterialShape, MaterialSpeed,
    MaterialText, MaterialTransition, MaterialVideo, MaterialVideoEffect, Materials, MotionBlur,
    RawMaterial, Reframe, Relight, RichFill, RichFont, RichPaint, RichShadow, RichSolid,
    RichStroke, RichStyle, RichText, SHAPE_ARROW, SHAPE_CONTOUR, SHAPE_FILL, SHAPE_LINE,
    SHAPE_LINE_WIDTH, SHAPE_PEN, SHAPE_SHADOW, SHAPE_SIZE, SHAPE_STROKE_WIDTH, ShapeKind,
    ShapePoint, SpeedPoint, SpeedPreset, StickerAnimation, SubtitleKeywords, TextRange,
    TimeMarkItem, VideoAlgorithm, VideoStable, VisibleBeat, VisibleMark, WordColumns,
    beat_timeline_time, blend_extents, blend_window, clip_mark_time, constant_target_duration,
    curve_speed_at, curve_speed_factor, curve_target_duration, keywords_in_ranges, new_beats,
    new_time_mark, referenced_time_marks, segment_beats, segment_bloom, segment_time_mark,
    set_time_mark_items, set_user_beats, source_time_at, split_curve, time_mark_items,
    track_holds_beats, user_beats, visible_beats, visible_marks,
};
pub use meta::MetaType;
pub use relationship::{LinkPair, Relationship};
pub use segment::{APPLY_ONE_CLIP, HdrSettings, ResponsiveLayout, Segment};
pub use shape_keys::{
    MAX_SHAPE_SHADOW_DISTANCE, MAX_SHAPE_SIZE, MAX_SHAPE_STROKE, SHAPE_STYLE_KEYS, ShapeStyleKey,
    shape_style_at,
};
pub use text_keys::{
    GLOW_RANGE_KEYS, GLOW_STRENGTH_KEYS, TEXT_STYLE_KEYS, TextStyleKey, glow_at, glow_keyed,
    text_style_at,
};
pub use time::{FrameError, FrameRate, Micros, RangeError, TimeRange, TimecodeLabel};
pub use track::Track;
pub use tracking::{TrackMode, TrackPath, TrackSample, VIDEO_TRACKING};
