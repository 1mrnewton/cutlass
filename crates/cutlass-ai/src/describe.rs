//! `describe_project`: the compact, deterministic timeline summary the
//! model reasons over.
//!
//! Pushed (never retrieved): the agent loop serializes a fresh
//! [`ProjectSummary`] + [`EditorContext`] into every prompt, and again into
//! tool results after edits, so the model always sees the world it is
//! editing. Output order is deterministic (stack order for tracks, start
//! order for clips, id order for media) so eval tests can assert verbatim.

use cutlass_models::{ClipSource, Generator, Project, Rational, Shape, Track};
use serde::{Deserialize, Serialize};

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
    /// Canvas settings (set_canvas). Omitted at the default (auto aspect,
    /// black background).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canvas: Option<CanvasSummary>,
    /// The media pool, id-ascending.
    pub media: Vec<MediaSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanvasSummary {
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectSummary {
    /// Catalog id, e.g. "gaussian_blur".
    pub effect: String,
    /// Current parameter values, sampled at the clip start, by name.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub params: std::collections::BTreeMap<String, f64>,
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
                    fade_in: (clip.fade_in > 0).then(|| seconds(clip.fade_in, rate)),
                    fade_out: (clip.fade_out > 0).then(|| seconds(clip.fade_out, rate)),
                    crop: (!clip.crop.is_full()).then(|| {
                        let c = clip.crop;
                        [
                            f64::from(c.x),
                            f64::from(c.y),
                            f64::from(1.0 - c.x - c.w),
                            f64::from(1.0 - c.y - c.h),
                        ]
                    }),
                    flip_h: clip.flip_h.then_some(true),
                    flip_v: clip.flip_v.then_some(true),
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

    let canvas = project.timeline().canvas();
    let canvas = (!canvas.is_default()).then(|| CanvasSummary {
        aspect: canvas.aspect.name().to_string(),
        background: canvas.background,
    });

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
mod tests {
    use super::*;
    use cutlass_models::{MediaSource, RationalTime, TimeRange, TrackKind};

    const R24: Rational = Rational::FPS_24;

    #[test]
    fn summary_is_deterministic_and_complete() {
        let mut project = Project::new("demo", R24);
        let media = project.add_media(MediaSource::new(
            "/footage/interview.mp4",
            1920,
            1080,
            R24,
            24 * 60,
            true,
        ));
        let video = project.add_track(TrackKind::Video, "V1");
        let text = project.add_track(TrackKind::Text, "Titles");

        // Insert out of timeline order to prove ordering is by start time.
        let late = project
            .add_clip(
                video,
                media,
                TimeRange::at_rate(0, 48, R24),
                RationalTime::new(96, R24),
            )
            .unwrap();
        let early = project
            .add_clip(
                video,
                media,
                TimeRange::at_rate(48, 48, R24),
                RationalTime::new(0, R24),
            )
            .unwrap();
        project
            .add_generated(
                text,
                Generator::text("INTRO"),
                TimeRange::at_rate(24, 48, R24),
            )
            .unwrap();

        let summary = summarize(&project);
        assert_eq!(summary.name, "demo");
        assert_eq!(summary.frame_rate_fps, 24.0);
        assert_eq!(summary.duration_seconds, 6.0);
        assert_eq!(summary.tracks.len(), 2);
        assert_eq!(summary.media.len(), 1);

        let v1 = &summary.tracks[0];
        assert_eq!(v1.kind, "video");
        let clip_ids: Vec<u64> = v1.clips.iter().map(|c| c.id).collect();
        assert_eq!(clip_ids, vec![early.raw(), late.raw()]);
        assert_eq!(v1.clips[0].start_seconds, 0.0);
        assert_eq!(v1.clips[0].duration_seconds, 2.0);
        assert_eq!(v1.clips[0].start_frames, 0);
        assert_eq!(v1.clips[0].duration_frames, 48);
        match &v1.clips[0].content {
            ClipContent::Media {
                file,
                source_start_seconds,
                ..
            } => {
                assert_eq!(file, "interview.mp4");
                assert_eq!(*source_start_seconds, 2.0);
            }
            other => panic!("expected media content, got {other:?}"),
        }

        let titles = &summary.tracks[1];
        assert_eq!(titles.kind, "text");
        assert_eq!(
            titles.clips[0].content,
            ClipContent::Text {
                text: "INTRO".to_string()
            }
        );

        assert_eq!(summary.media[0].file, "interview.mp4");
        assert_eq!(summary.media[0].duration_seconds, 60.0);
        assert!(summary.media[0].has_audio);
    }

    #[test]
    fn phantom_generators_surface_as_other() {
        let mut project = Project::new("phantoms", R24);
        let adj = project.add_track(TrackKind::Adjustment, "FX");
        project
            .add_generated(adj, Generator::Adjustment, TimeRange::at_rate(0, 24, R24))
            .unwrap();

        let summary = summarize(&project);
        assert_eq!(summary.tracks[0].kind, "adjustment");
        assert_eq!(
            summary.tracks[0].clips[0].content,
            ClipContent::Other {
                kind: "adjustment".to_string()
            }
        );
    }

    #[test]
    fn canvas_surfaces_only_when_not_default() {
        let mut project = Project::new("canvas", R24);
        assert_eq!(summarize(&project).canvas, None);

        project
            .timeline_mut()
            .set_canvas(cutlass_models::CanvasSettings {
                aspect: cutlass_models::CanvasAspect::Tall9x16,
                background: [20, 20, 28],
            });
        assert_eq!(
            summarize(&project).canvas,
            Some(CanvasSummary {
                aspect: "9:16".to_string(),
                background: [20, 20, 28],
            })
        );
    }

    #[test]
    fn summary_and_context_serialize_to_stable_json() {
        let mut project = Project::new("json", R24);
        let track = project.add_track(TrackKind::Text, "T");
        project
            .add_generated(track, Generator::text("hi"), TimeRange::at_rate(0, 24, R24))
            .unwrap();

        let summary_json = serde_json::to_value(summarize(&project)).unwrap();
        let clip = &summary_json["tracks"][0]["clips"][0];
        assert_eq!(clip["content"], "text");
        assert_eq!(clip["text"], "hi");

        let ctx = EditorContext {
            selected_clips: vec![12],
            playhead_seconds: 3.5,
            in_point_seconds: None,
            out_point_seconds: None,
        };
        let ctx_json = serde_json::to_value(&ctx).unwrap();
        assert_eq!(
            ctx_json,
            serde_json::json!({ "selected_clips": [12], "playhead_seconds": 3.5 })
        );
    }
}
