//! `describe_project`: the compact, deterministic timeline summary the
//! model reasons over.
//!
//! Pushed (never retrieved): the agent loop serializes a fresh
//! [`ProjectSummary`] + [`EditorContext`] into every prompt, and again into
//! tool results after edits, so the model always sees the world it is
//! editing. Output order is deterministic (stack order for tracks, start
//! order for clips, id order for media) so eval tests can assert verbatim.
//!
//! Keyframed clip params appear under [`ClipSummary::keyframes`] as compact
//! `{t,v,e}` points: `t` is absolute timeline seconds (same as
//! `set_param_keyframe.at`), `v` is the wire value shape, and `e` is the wire
//! easing name (omitted when linear). A param with keyframes omits its static
//! field on the clip summary.

use std::collections::BTreeMap;

use cutlass_models::{
    Clip, ClipSource, Easing, Generator, MediaKind, Param, Project, Rational, Scale2, Shape, Track,
};
use serde::{Deserialize, Serialize};

use crate::wire::WireScale;

/// Vertical reference used by the renderer for fixed aspect presets (1080p
/// baseline on the longer side). Kept in sync with `cutlass_render::canvas_size`.
const CANVAS_REFERENCE_HEIGHT: f32 = 1080.0;
/// Fallback when aspect is Auto and no video media is on a visual track.
const DEFAULT_CANVAS_PIXELS: (u32, u32) = (1920, 1080);

/// UI session state captured when the user hits send. This is how "the
/// selected clip" and "at the playhead" resolve to ids and times.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EditorContext {
    /// Ids of the clips currently selected on the timeline.
    pub selected_clips: Vec<u64>,
    /// Playhead position in seconds.
    pub playhead_seconds: f64,
    /// Loop/range in-point in seconds, if one is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub in_point_seconds: Option<f64>,
    /// Loop/range out-point in seconds, if one is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_point_seconds: Option<f64>,
}

/// Token-bounded snapshot of the whole project.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectSummary {
    pub name: String,
    /// Timeline frame rate in frames per second.
    pub frame_rate_fps: f64,
    /// End of the last clip on any track, in seconds.
    pub duration_seconds: f64,
    /// Tracks in stack order, bottom (composited first) to top.
    pub tracks: Vec<TrackSummary>,
    /// Ruler markers in tick order (M1). Omitted when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub markers: Vec<MarkerSummary>,
    /// Canvas size and settings (set_canvas). Always present so the model
    /// knows the pixel frame placement fractions refer to.
    pub canvas: CanvasSummary,
    /// The media pool, id-ascending.
    pub media: Vec<MediaSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanvasSummary {
    /// Resolved canvas width in pixels (same box preview/export use).
    pub width: u32,
    /// Resolved canvas height in pixels.
    pub height: u32,
    /// Aspect preset name: auto, 16:9, 9:16, 1:1, 4:5, 21:9.
    pub aspect: String,
    /// Background color as `[red, green, blue]`, each 0-255.
    pub background: [u8; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarkerSummary {
    pub id: u64,
    /// Timeline position in seconds.
    pub at_seconds: f64,
    /// Exact timeline position in frames at the project rate.
    pub at_frames: i64,
    /// Short label; omitted when empty.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    /// Palette name: teal, blue, purple, pink, red, orange, yellow, green.
    pub color: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackSummary {
    pub id: u64,
    /// Lane kind: video, audio, text, sticker, effect, filter, adjustment.
    pub kind: String,
    pub name: String,
    pub enabled: bool,
    pub muted: bool,
    pub locked: bool,
    /// The permanent main video track (CapCut's magnetic lane). It cannot be
    /// removed; every other visual lane stacks above it. Omitted elsewhere.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub main: bool,
    /// Clips in timeline order.
    pub clips: Vec<ClipSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipSummary {
    pub id: u64,
    /// Timeline start in seconds.
    pub start_seconds: f64,
    /// Clip length in seconds.
    pub duration_seconds: f64,
    /// Exact timeline start in frames at the project rate.
    pub start_frames: i64,
    /// Exact clip length in frames at the project rate.
    pub duration_frames: i64,
    #[serde(flatten)]
    pub content: ClipContent,
    /// Link group id; clips sharing one move/trim together.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<u64>,
    /// Playback rate multiplier (set_clip_speed); absent when 1x.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    /// Playing backwards (set_clip_speed); absent when forward.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reversed: Option<bool>,
    /// Carries a varying-speed ramp (set_speed_curve); absent when the clip
    /// plays at a single constant speed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed_ramp: Option<bool>,
    /// Pitch rides the playback speed instead of being preserved
    /// (set_clip_pitch); absent in the default pitch-locked state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch_follows_speed: Option<bool>,
    /// Audio gain multiplier (set_clip_audio); absent when 1.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volume: Option<f64>,
    /// Stereo pan (−1 left … +1 right); absent when centered (0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pan: Option<f64>,
    /// Fade-in seconds (set_clip_audio); absent when 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fade_in: Option<f64>,
    /// Fade-out seconds (set_clip_audio); absent when 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fade_out: Option<f64>,
    /// Fractions trimmed off each edge as `[left, top, right, bottom]`
    /// (set_clip_crop); absent when the full frame shows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crop: Option<[f64; 4]>,
    /// Mirrored left-right (set_clip_crop); absent when not flipped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flip_h: Option<bool>,
    /// Mirrored top-bottom (set_clip_crop); absent when not flipped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flip_v: Option<bool>,
    /// Anchor offset from the canvas CENTER in canvas-width/height fractions
    /// (+x right, +y down); `[0,0]` = centered (set_clip_transform). Absent
    /// when centered. Omitted entirely when the position param is keyframed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<[f64; 2]>,
    /// Pivot within content bounds (0 = left/top, 0.5 = center). Absent at
    /// `[0.5, 0.5]`. Omitted when the anchor_point param is keyframed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<[f64; 2]>,
    /// Placement scale (set_clip_transform): a bare number when uniform, or
    /// `[x, y]` when split. Absent at the default identity (1). Omitted when
    /// the scale param is keyframed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<WireScale>,
    /// Clockwise rotation in degrees about the anchor (set_clip_transform).
    /// Absent at 0. Omitted when the rotation param is keyframed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation: Option<f64>,
    /// Layer opacity 0.0–1.0 (set_clip_transform). Absent at 1.0. Omitted
    /// when the opacity param is keyframed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
    /// Keyframed clip params, keyed by wire name (`position`, `anchor_point`,
    /// `scale`, `rotation`, `opacity`, `volume`, `pan`). Absent when nothing
    /// is animated. Each point uses `t`/`v`/`e` (absolute timeline seconds,
    /// wire-shaped value, wire easing name).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keyframes: Option<BTreeMap<String, Vec<KeyframeSummary>>>,
    /// Visual effects in chain order (add_effect); the index of each entry is
    /// what remove_effect / move_effect / set_effect_param address. Absent
    /// when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<EffectSummary>,
    /// Transition at this clip's right cut (add_transition), if any. The
    /// catalog id of the blend into the next clip.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transition: Option<String>,
    /// Mask kind plus non-default geometry (set_clip_mask); absent when no mask.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<String>,
    /// Filter preset id (set_clip_filter); absent when none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
    /// Non-neutral manual adjust sliders (set_clip_adjustments), e.g.
    /// "tint=0.25, sharpness=0.50"; absent when every slider is neutral.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adjust: Option<String>,
    /// Blend mode id (set_clip_blend_mode); absent when normal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blend: Option<String>,
    /// Motion blur summary when enabled (set_motion_blur), e.g.
    /// "shutter=180 samples=8"; absent when off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub motion_blur: Option<String>,
    /// Active layer-style blocks (set_layer_styles), e.g. "shadow, outline";
    /// absent when none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub styles: Option<String>,
    /// Entrance animation id (set_clip_animation in slot); absent when none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animation_in: Option<String>,
    /// Exit animation id; absent when none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animation_out: Option<String>,
    /// Combo animation id; absent when none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animation_combo: Option<String>,
    /// Audio role tag (set_audio_role); absent when untagged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_role: Option<String>,
}

/// One keyframe on a clip param, in the same units the model writes with
/// `set_param_keyframe` / `remove_param_keyframe`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyframeSummary {
    /// Absolute timeline seconds (clip start + clip-relative tick). Matches
    /// `set_param_keyframe.at` so existing keyframes can be addressed directly.
    #[serde(rename = "t")]
    pub at: f64,
    /// Wire-shaped value: `[x,y]` for vec2, bare number for scalars, uniform
    /// number or `[x,y]` for scale ([`WireScale`]).
    #[serde(rename = "v")]
    pub value: serde_json::Value,
    /// Wire easing name (`ease_out`, `snappy`, `hold`, …). Omitted when linear.
    #[serde(rename = "e", default, skip_serializing_if = "Option::is_none")]
    pub easing: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectSummary {
    /// Catalog id, e.g. "gaussian_blur".
    pub effect: String,
    /// Current parameter values, sampled at the clip start, by name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub params: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "content", rename_all = "snake_case")]
pub enum ClipContent {
    /// A trimmed range of an imported media file.
    Media {
        media: u64,
        file: String,
        source_start_seconds: f64,
        source_duration_seconds: f64,
    },
    Text {
        text: String,
    },
    Solid {
        rgba: [u8; 4],
    },
    Shape {
        shape: String,
        rgba: [u8; 4],
        width: f64,
        height: f64,
    },
    /// A generator kind the agent cannot create or edit.
    Other {
        kind: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MediaSummary {
    pub id: u64,
    pub file: String,
    pub duration_seconds: f64,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub has_audio: bool,
}

fn seconds(ticks: i64, rate: Rational) -> f64 {
    ticks as f64 * rate.seconds_per_unit()
}

/// Pixel size of the project's canvas — mirrors `cutlass_render::canvas_size`
/// so describe stays free of a render dependency.
fn describe_canvas_size(project: &Project) -> (u32, u32) {
    match project.timeline().canvas().aspect.ratio() {
        Some((rw, rh)) => ratio_to_pixels(rw, rh),
        None => auto_canvas_size(project),
    }
}

fn ratio_to_pixels(rw: u32, rh: u32) -> (u32, u32) {
    let (rw, rh) = (rw as f32, rh as f32);
    let (w, h) = if rw >= rh {
        (
            (CANVAS_REFERENCE_HEIGHT * rw / rh).round(),
            CANVAS_REFERENCE_HEIGHT,
        )
    } else {
        (
            CANVAS_REFERENCE_HEIGHT,
            (CANVAS_REFERENCE_HEIGHT * rh / rw).round(),
        )
    };
    (even_pixels(w as u32), even_pixels(h as u32))
}

fn auto_canvas_size(project: &Project) -> (u32, u32) {
    let mut best: Option<(u32, u32)> = None;
    for track in project.timeline().tracks_ordered() {
        if !track.kind.is_visual() {
            continue;
        }
        for clip in track.clips() {
            let Some(id) = clip.media() else { continue };
            let Some(media) = project.media(id) else {
                continue;
            };
            if media.kind() != MediaKind::Video {
                continue;
            }
            let area = u64::from(media.width) * u64::from(media.height);
            if best.is_none_or(|(bw, bh)| area > u64::from(bw) * u64::from(bh)) {
                best = Some((media.width, media.height));
            }
        }
    }
    best.map_or(DEFAULT_CANVAS_PIXELS, |(w, h)| {
        (even_pixels(w), even_pixels(h))
    })
}

fn even_pixels(v: u32) -> u32 {
    (v & !1).max(2)
}

fn wire_easing_name(easing: Easing) -> Option<String> {
    match easing {
        Easing::Linear => None,
        Easing::EaseIn => Some("ease_in".into()),
        Easing::EaseOut => Some("ease_out".into()),
        Easing::EaseInOut => Some("ease_in_out".into()),
        Easing::Hold => Some("hold".into()),
        Easing::Bezier { .. } => Some(easing.preset_id().unwrap_or("bezier").into()),
    }
}

fn scale_wire_value(s: Scale2) -> serde_json::Value {
    if s.is_uniform() {
        serde_json::json!(f64::from(s.x))
    } else {
        serde_json::json!([f64::from(s.x), f64::from(s.y)])
    }
}

fn push_keyframes<T, F>(
    out: &mut BTreeMap<String, Vec<KeyframeSummary>>,
    name: &str,
    param: &Param<T>,
    clip_start_ticks: i64,
    rate: Rational,
    mut value_json: F,
) where
    T: Copy,
    F: FnMut(T) -> serde_json::Value,
{
    if !param.is_animated() {
        return;
    }
    let points: Vec<KeyframeSummary> = param
        .keyframes()
        .iter()
        .map(|kf| KeyframeSummary {
            at: seconds(clip_start_ticks + kf.tick, rate),
            value: value_json(kf.value),
            easing: wire_easing_name(kf.easing),
        })
        .collect();
    if !points.is_empty() {
        out.insert(name.to_string(), points);
    }
}

fn summarize_clip_keyframes(
    clip: &Clip,
    rate: Rational,
) -> Option<BTreeMap<String, Vec<KeyframeSummary>>> {
    let start = clip.timeline.start.value;
    let mut map = BTreeMap::new();
    push_keyframes(
        &mut map,
        "position",
        &clip.transform.position,
        start,
        rate,
        |v| serde_json::json!([f64::from(v[0]), f64::from(v[1])]),
    );
    push_keyframes(
        &mut map,
        "anchor_point",
        &clip.transform.anchor_point,
        start,
        rate,
        |v| serde_json::json!([f64::from(v[0]), f64::from(v[1])]),
    );
    push_keyframes(
        &mut map,
        "scale",
        &clip.transform.scale,
        start,
        rate,
        scale_wire_value,
    );
    push_keyframes(
        &mut map,
        "rotation",
        &clip.transform.rotation,
        start,
        rate,
        |v| serde_json::json!(f64::from(v)),
    );
    push_keyframes(
        &mut map,
        "opacity",
        &clip.transform.opacity,
        start,
        rate,
        |v| serde_json::json!(f64::from(v)),
    );
    push_keyframes(&mut map, "volume", &clip.volume, start, rate, |v| {
        serde_json::json!(f64::from(v))
    });
    push_keyframes(&mut map, "pan", &clip.pan, start, rate, |v| {
        serde_json::json!(f64::from(v))
    });
    (!map.is_empty()).then_some(map)
}

fn summarize_adjust(adjust: &cutlass_models::ColorAdjustments) -> Option<String> {
    if adjust.is_neutral() {
        return None;
    }
    let mut parts = Vec::new();
    let push = |parts: &mut Vec<String>, name: &str, param: &cutlass_models::Param<f32>| {
        if let Some(v) = param.constant()
            && v != 0.0
        {
            parts.push(format!("{name}={v:.2}"));
        } else if !matches!(param, cutlass_models::Param::Constant(_)) {
            parts.push(format!("{name}=kf"));
        }
    };
    push(&mut parts, "brightness", &adjust.brightness);
    push(&mut parts, "contrast", &adjust.contrast);
    push(&mut parts, "saturation", &adjust.saturation);
    push(&mut parts, "exposure", &adjust.exposure);
    push(&mut parts, "temperature", &adjust.temperature);
    push(&mut parts, "tint", &adjust.tint);
    push(&mut parts, "hue", &adjust.hue);
    push(&mut parts, "highlights", &adjust.highlights);
    push(&mut parts, "shadows", &adjust.shadows);
    push(&mut parts, "sharpness", &adjust.sharpness);
    push(&mut parts, "vignette", &adjust.vignette);
    (!parts.is_empty()).then(|| parts.join(", "))
}

fn track_kind_name(track: &Track) -> &'static str {
    match track.kind {
        cutlass_models::TrackKind::Video => "video",
        cutlass_models::TrackKind::Audio => "audio",
        cutlass_models::TrackKind::Text => "text",
        cutlass_models::TrackKind::Sticker => "sticker",
        cutlass_models::TrackKind::Effect => "effect",
        cutlass_models::TrackKind::Filter => "filter",
        cutlass_models::TrackKind::Adjustment => "adjustment",
    }
}

fn clip_content(project: &Project, content: &ClipSource) -> ClipContent {
    match content {
        ClipSource::Media { media, source } => {
            let (file, rate) = project
                .media(*media)
                .map(|m| (file_name(m.path()), m.frame_rate))
                .unwrap_or_else(|| ("<missing>".to_string(), Rational::FPS_24));
            ClipContent::Media {
                media: media.raw(),
                file,
                source_start_seconds: seconds(source.start.value, rate),
                source_duration_seconds: seconds(source.duration.value, rate),
            }
        }
        ClipSource::Generated(generator) => match generator {
            Generator::Text { content, .. } => ClipContent::Text {
                text: content.clone(),
            },
            Generator::SolidColor { rgba } => ClipContent::Solid { rgba: *rgba },
            Generator::Shape {
                shape,
                rgba,
                width,
                height,
                ..
            } => ClipContent::Shape {
                shape: match shape {
                    Shape::Rectangle => "rectangle".to_string(),
                    Shape::Ellipse => "ellipse".to_string(),
                    Shape::Polygon { sides } => format!("polygon({sides})"),
                    Shape::Star { points, .. } => format!("star({points})"),
                    Shape::Line => "line".to_string(),
                    Shape::Arrow => "arrow".to_string(),
                    Shape::Heart => "heart".to_string(),
                    Shape::Path(_) => "path".to_string(),
                },
                rgba: rgba.sample(0),
                width: f64::from(width.sample(0)),
                height: f64::from(height.sample(0)),
            },
            Generator::Sticker { asset } => ClipContent::Other {
                kind: if asset.is_empty() {
                    "sticker".to_string()
                } else {
                    format!("sticker:{asset}")
                },
            },
            Generator::Lottie { path, .. } => ClipContent::Other {
                kind: format!(
                    "lottie:{}",
                    std::path::Path::new(path)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("animation")
                ),
            },
            Generator::Effect => ClipContent::Other {
                kind: "effect".to_string(),
            },
            Generator::Filter => ClipContent::Other {
                kind: "filter".to_string(),
            },
            Generator::Adjustment => ClipContent::Other {
                kind: "adjustment".to_string(),
            },
        },
    }
}

fn file_name(path: &std::path::Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

/// Build the model-facing summary of `project`.
pub fn summarize(project: &Project) -> ProjectSummary {
    let rate = project.timeline().frame_rate;

    let tracks: Vec<TrackSummary> = project
        .timeline()
        .tracks_ordered()
        .map(|track| TrackSummary {
            id: track.id.raw(),
            kind: track_kind_name(track).to_string(),
            name: track.name.clone(),
            enabled: track.enabled,
            muted: track.muted,
            locked: track.locked,
            main: track.main,
            clips: track
                .clips_ordered()
                .into_iter()
                .map(|clip| ClipSummary {
                    id: clip.id.raw(),
                    start_seconds: seconds(clip.timeline.start.value, rate),
                    duration_seconds: seconds(clip.timeline.duration.value, rate),
                    start_frames: clip.timeline.start.value,
                    duration_frames: clip.timeline.duration.value,
                    content: clip_content(project, &clip.content),
                    link: clip.link.map(|l| l.raw()),
                    speed: (clip.speed.num != clip.speed.den)
                        .then(|| f64::from(clip.speed.num) / f64::from(clip.speed.den)),
                    reversed: clip.reversed.then_some(true),
                    speed_ramp: clip.has_speed_curve().then_some(true),
                    pitch_follows_speed: (!clip.preserve_pitch).then_some(true),
                    volume: clip.volume.constant().filter(|v| *v != 1.0).map(f64::from),
                    pan: clip.pan.constant().filter(|v| *v != 0.0).map(f64::from),
                    fade_in: (clip.fade_in > 0).then(|| seconds(clip.fade_in, rate)),
                    fade_out: (clip.fade_out > 0).then(|| seconds(clip.fade_out, rate)),
                    crop: {
                        // Constants: describe the stored framing. Keyframed:
                        // sample at clip-relative 0 (playhead-aware describe
                        // is a later pass).
                        let c = clip.crop.sample(0);
                        (!c.is_full() || clip.crop.is_animated()).then(|| {
                            [
                                f64::from(c.x),
                                f64::from(c.y),
                                f64::from(1.0 - c.x - c.w),
                                f64::from(1.0 - c.y - c.h),
                            ]
                        })
                    },
                    flip_h: clip.flip_h.then_some(true),
                    flip_v: clip.flip_v.then_some(true),
                    position: {
                        // Static sample at clip-relative 0. Animated params
                        // omit the static field (keyframe dump is the truth).
                        (!clip.transform.position.is_animated())
                            .then(|| {
                                let p = clip.transform.position.sample(0);
                                (p != [0.0, 0.0]).then_some([f64::from(p[0]), f64::from(p[1])])
                            })
                            .flatten()
                    },
                    anchor: {
                        (!clip.transform.anchor_point.is_animated())
                            .then(|| {
                                let a = clip.transform.anchor_point.sample(0);
                                (a != [0.5, 0.5]).then_some([f64::from(a[0]), f64::from(a[1])])
                            })
                            .flatten()
                    },
                    scale: {
                        (!clip.transform.scale.is_animated())
                            .then(|| {
                                let s = clip.transform.scale.sample(0);
                                (s != Scale2::uniform(1.0)).then(|| {
                                    if s.is_uniform() {
                                        WireScale::Uniform(f64::from(s.x))
                                    } else {
                                        WireScale::Axes([f64::from(s.x), f64::from(s.y)])
                                    }
                                })
                            })
                            .flatten()
                    },
                    rotation: {
                        (!clip.transform.rotation.is_animated())
                            .then(|| {
                                let r = clip.transform.rotation.sample(0);
                                (r != 0.0).then_some(f64::from(r))
                            })
                            .flatten()
                    },
                    opacity: {
                        (!clip.transform.opacity.is_animated())
                            .then(|| {
                                let o = clip.transform.opacity.sample(0);
                                (o != 1.0).then_some(f64::from(o))
                            })
                            .flatten()
                    },
                    keyframes: summarize_clip_keyframes(clip, rate),
                    effects: clip
                        .effects
                        .iter()
                        .map(|fx| EffectSummary {
                            effect: fx.effect_id.clone(),
                            params: fx
                                .spec()
                                .map(|spec| {
                                    spec.params
                                        .iter()
                                        .filter_map(|p| {
                                            fx.sample_param(p.name, 0.0)
                                                .map(|v| (p.name.to_string(), f64::from(v)))
                                        })
                                        .collect()
                                })
                                .unwrap_or_default(),
                        })
                        .collect(),
                    transition: track
                        .transition_at(clip.id)
                        .map(|t| t.transition_id.clone()),
                    mask: clip.mask.as_ref().map(|m| {
                        let kind = match m.kind {
                            cutlass_models::MaskKind::Linear => "linear",
                            cutlass_models::MaskKind::Mirror => "mirror",
                            cutlass_models::MaskKind::Circle => "circle",
                            cutlass_models::MaskKind::Rectangle => "rectangle",
                            cutlass_models::MaskKind::Heart => "heart",
                            cutlass_models::MaskKind::Star => "star",
                        };
                        let mut parts = vec![kind.to_string()];
                        if let Some(c) = m.center.constant()
                            && c != [0.0, 0.0]
                        {
                            parts.push(format!("center=[{:.2}, {:.2}]", c[0], c[1]));
                        }
                        if let Some(s) = m.size.constant()
                            && s != [1.0, 1.0]
                        {
                            parts.push(format!("size=[{:.2}, {:.2}]", s[0], s[1]));
                        }
                        if let Some(r) = m.rotation.constant()
                            && r != 0.0
                        {
                            parts.push(format!("rot={r:.0}"));
                        }
                        if let Some(r) = m.roundness.constant()
                            && r != 0.0
                        {
                            parts.push(format!("round={r:.2}"));
                        }
                        parts.join(" ")
                    }),
                    filter: clip.filter.as_ref().map(|f| f.id.clone()),
                    adjust: summarize_adjust(&clip.adjust),
                    blend: (!clip.blend_mode.is_normal()).then(|| clip.blend_mode.id().to_string()),
                    motion_blur: clip.motion_blur.enabled.then(|| {
                        format!(
                            "shutter={} samples={}",
                            clip.motion_blur.shutter_deg, clip.motion_blur.samples
                        )
                    }),
                    styles: {
                        let mut blocks = Vec::new();
                        if clip.styles.shadow.is_some() {
                            blocks.push("shadow");
                        }
                        if clip.styles.glow.is_some() {
                            blocks.push("glow");
                        }
                        if clip.styles.outline.is_some() {
                            blocks.push("outline");
                        }
                        if clip.styles.background.is_some() {
                            blocks.push("background");
                        }
                        (!blocks.is_empty()).then(|| blocks.join(", "))
                    },
                    animation_in: clip.animation_in.as_ref().map(|a| a.id.clone()),
                    animation_out: clip.animation_out.as_ref().map(|a| a.id.clone()),
                    animation_combo: clip.animation_combo.as_ref().map(|a| a.id.clone()),
                    audio_role: clip.audio_role.map(|r| r.id().to_string()),
                })
                .collect(),
        })
        .collect();

    let duration_ticks = project
        .timeline()
        .tracks_ordered()
        .map(Track::content_end)
        .max()
        .unwrap_or(0);

    let mut media: Vec<MediaSummary> = project
        .media_iter()
        .map(|m| MediaSummary {
            id: m.id.raw(),
            file: file_name(m.path()),
            duration_seconds: seconds(m.duration.value, m.frame_rate),
            width: m.width,
            height: m.height,
            fps: m.frame_rate.as_f64(),
            has_audio: m.has_audio,
        })
        .collect();
    media.sort_by_key(|m| m.id);

    let markers = project
        .timeline()
        .markers()
        .iter()
        .map(|m| MarkerSummary {
            id: m.id.raw(),
            at_seconds: seconds(m.tick.value, rate),
            at_frames: m.tick.value,
            name: m.name.clone(),
            color: m.color.name().to_string(),
        })
        .collect();

    let settings = project.timeline().canvas();
    let (width, height) = describe_canvas_size(project);
    let canvas = CanvasSummary {
        width,
        height,
        aspect: settings.aspect.name().to_string(),
        background: settings.background,
    };

    ProjectSummary {
        name: project.name.clone(),
        frame_rate_fps: rate.as_f64(),
        duration_seconds: seconds(duration_ticks, rate),
        tracks,
        markers,
        canvas,
        media,
    }
}

#[cfg(test)]
mod tests;
