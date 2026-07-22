//! Inspector helpers: resolve the selected clip for the property sheet, and
//! sample its animated transform at the playhead for the keyframe UI.

use crate::params::{
    apply_sampled_transform, row_state, sampled_scalar_param, sampled_transform,
    sampled_vec2_param, sampled_volume,
};
use crate::placement::position_preserving_center;
use crate::preview_select::{canvas_config, clip_placement};
use crate::{
    AudioSample, Clip, ClipAdjust, CompensatedPosition, ScalarParamSample, SelectedClipInfo,
    Sequence, TextClipStyle, TrackKind, TransformSample,
};
use cutlass_models::{
    TextAlignH, TextAlignV, TextBackground, TextCase, TextShadow, TextStroke,
    TextStyle as ModelTextStyle,
};
use slint::Model;

/// Convert the inspector's Slint [`ClipAdjust`] into the engine model.
pub fn adjust_from_ui(adjust: &ClipAdjust) -> cutlass_models::ColorAdjustments {
    cutlass_models::ColorAdjustments {
        brightness: adjust.brightness.into(),
        contrast: adjust.contrast.into(),
        saturation: adjust.saturation.into(),
        exposure: adjust.exposure.into(),
        temperature: adjust.temperature.into(),
        tint: adjust.tint.into(),
        hue: adjust.hue.into(),
        highlights: adjust.highlights.into(),
        shadows: adjust.shadows.into(),
        sharpness: adjust.sharpness.into(),
        vignette: adjust.vignette.into(),
    }
}

/// Convert the inspector's Slint `TextClipStyle` into the engine model.
///
/// The inverse of `projection::text_style_to_ui`: effect opacity (a separate
/// 0..=1 control) is folded back into the rgba alpha, and the disabled flags
/// collapse to `None`. The inspector always sends the complete style, so the
/// engine writes one coherent `Generator::Text`.
pub fn text_style_from_ui(style: &TextClipStyle) -> ModelTextStyle {
    let rgba = |c: slint::Color| [c.red(), c.green(), c.blue(), c.alpha()];
    let rgb_alpha = |c: slint::Color, a: f32| {
        [
            c.red(),
            c.green(),
            c.blue(),
            (a.clamp(0.0, 1.0) * 255.0).round() as u8,
        ]
    };
    ModelTextStyle {
        font: style.font.to_string(),
        size: style.size.into(),
        bold: style.bold,
        italic: style.italic,
        underline: style.underline,
        case: text_case_from_int(style.case),
        fill: rgba(style.fill).into(),
        letter_spacing: style.letter_spacing.into(),
        line_spacing: style.line_spacing.into(),
        align_h: align_h_from_int(style.align_h),
        align_v: align_v_from_int(style.align_v),
        wrap: style.wrap,
        stroke: style.stroke_enabled.then(|| TextStroke {
            rgba: rgba(style.stroke_color).into(),
            width: style.stroke_width.into(),
        }),
        background: style.background_enabled.then(|| TextBackground {
            rgba: rgb_alpha(style.background_color, style.background_opacity),
            radius: style.background_radius,
        }),
        shadow: style.shadow_enabled.then(|| TextShadow {
            rgba: rgb_alpha(style.shadow_color, style.shadow_opacity).into(),
            blur: style.shadow_blur.into(),
            distance: style.shadow_distance.into(),
        }),
        // The vendored inspector has no preset chips yet; a manual commit
        // carries the baked treatments above and drops the preset tag.
        effect_preset: None,
    }
}

fn text_case_from_int(case: i32) -> TextCase {
    match case {
        1 => TextCase::Upper,
        2 => TextCase::Lower,
        3 => TextCase::Title,
        _ => TextCase::Normal,
    }
}

fn align_h_from_int(align: i32) -> TextAlignH {
    match align {
        0 => TextAlignH::Left,
        2 => TextAlignH::Right,
        _ => TextAlignH::Center,
    }
}

fn align_v_from_int(align: i32) -> TextAlignV {
    match align {
        0 => TextAlignV::Top,
        2 => TextAlignV::Bottom,
        _ => TextAlignV::Middle,
    }
}

/// The inspector's per-playhead view of a clip's transform: every property
/// sampled at the (clamped) playhead, plus the keyframe row state driving
/// each row's diamond cluster. Pure — re-evaluated by Slint when the
/// playhead or the projected clip changes, so value rows track playback
/// without a projection republish per tick.
pub fn sample_transform(clip: &Clip, playhead: i32) -> TransformSample {
    let t = sampled_transform(clip, playhead);
    TransformSample {
        position_x: t.position[0],
        position_y: t.position[1],
        anchor_x: t.anchor_point[0],
        anchor_y: t.anchor_point[1],
        scale: t.scale,
        rotation: t.rotation,
        opacity: t.opacity,
        position_row: row_state(&clip.kf_position, playhead),
        anchor_row: row_state(&clip.kf_anchor, playhead),
        scale_row: row_state(&clip.kf_scale, playhead),
        rotation_row: row_state(&clip.kf_rotation, playhead),
        opacity_row: row_state(&clip.kf_opacity, playhead),
    }
}

/// Sample a scalar text-style or look curve at the playhead for inspector
/// rows. Unknown ids are defensive no-ops; the UI only supplies known keys.
pub fn sample_scalar_param(clip: &Clip, param: &str, playhead: i32) -> ScalarParamSample {
    // Axis display keys for shared vec2 params — same row-state precedent as
    // transform `position` X/Y.
    if param == "style_shadow_offset_x" || param == "style_shadow_offset_y" {
        let offset =
            sampled_vec2_param(clip, "style_shadow_offset", playhead).unwrap_or([0.0, 0.0]);
        return ScalarParamSample {
            value: if param.ends_with("_x") {
                offset[0]
            } else {
                offset[1]
            },
            row: row_state(&clip.kf_style_shadow_offset, playhead),
        };
    }
    if param == "look_mask_center_x" || param == "look_mask_center_y" {
        let center = sampled_vec2_param(clip, "look_mask_center", playhead).unwrap_or([0.0, 0.0]);
        return ScalarParamSample {
            value: if param.ends_with("_x") {
                center[0]
            } else {
                center[1]
            },
            row: row_state(&clip.kf_look_mask_center, playhead),
        };
    }
    if param == "look_mask_size_x" || param == "look_mask_size_y" {
        let size = sampled_vec2_param(clip, "look_mask_size", playhead).unwrap_or([1.0, 1.0]);
        return ScalarParamSample {
            value: if param.ends_with("_x") {
                size[0]
            } else {
                size[1]
            },
            row: row_state(&clip.kf_look_mask_size, playhead),
        };
    }
    let keyframes = match param {
        "text_size" => &clip.kf_text_size,
        "text_letter_spacing" => &clip.kf_text_letter_spacing,
        "text_line_spacing" => &clip.kf_text_line_spacing,
        "text_stroke_width" => &clip.kf_text_stroke_width,
        "text_shadow_blur" => &clip.kf_text_shadow_blur,
        "text_shadow_distance" => &clip.kf_text_shadow_distance,
        "look_filter_intensity" => &clip.kf_look_filter_intensity,
        "look_lut_intensity" => &clip.kf_look_lut_intensity,
        "look_adjust_brightness" => &clip.kf_look_adjust_brightness,
        "look_adjust_contrast" => &clip.kf_look_adjust_contrast,
        "look_adjust_saturation" => &clip.kf_look_adjust_saturation,
        "look_adjust_exposure" => &clip.kf_look_adjust_exposure,
        "look_adjust_temperature" => &clip.kf_look_adjust_temperature,
        "look_adjust_tint" => &clip.kf_look_adjust_tint,
        "look_adjust_hue" => &clip.kf_look_adjust_hue,
        "look_adjust_highlights" => &clip.kf_look_adjust_highlights,
        "look_adjust_shadows" => &clip.kf_look_adjust_shadows,
        "look_adjust_sharpness" => &clip.kf_look_adjust_sharpness,
        "look_adjust_vignette" => &clip.kf_look_adjust_vignette,
        "style_shadow_blur" => &clip.kf_style_shadow_blur,
        "style_glow_radius" => &clip.kf_style_glow_radius,
        "style_glow_intensity" => &clip.kf_style_glow_intensity,
        "style_outline_width" => &clip.kf_style_outline_width,
        "style_background_padding" => &clip.kf_style_background_padding,
        "style_background_radius" => &clip.kf_style_background_radius,
        "look_mask_feather" => &clip.kf_look_mask_feather,
        "look_mask_rotation" => &clip.kf_look_mask_rotation,
        "look_mask_roundness" => &clip.kf_look_mask_roundness,
        "look_chroma_strength" => &clip.kf_look_chroma_strength,
        "look_chroma_shadow" => &clip.kf_look_chroma_shadow,
        _ => {
            return ScalarParamSample {
                value: 0.0,
                row: Default::default(),
            };
        }
    };
    ScalarParamSample {
        value: sampled_scalar_param(clip, param, playhead).unwrap_or_default(),
        row: row_state(keyframes, playhead),
    }
}

/// Position that keeps the composited frame fixed while the in-content
/// anchor moves — mirrors the preview anchor-handle gesture.
pub fn compensate_anchor_position(
    clip: &Clip,
    sequence: Sequence,
    playhead: i32,
    anchor_x: f32,
    anchor_y: f32,
    scale: f32,
    rotation: f32,
) -> CompensatedPosition {
    let canvas = canvas_config(&sequence);
    let mut c = clip.clone();
    apply_sampled_transform(&mut c, playhead);
    c.transform_scale = scale;
    c.transform_rotation = rotation;
    let placement = clip_placement(&c, &canvas);
    let position = position_preserving_center(
        placement.center,
        placement.size,
        [anchor_x, anchor_y],
        rotation,
        &canvas,
    );
    CompensatedPosition {
        position_x: position[0],
        position_y: position[1],
    }
}

/// The inspector's per-playhead view of a clip's audio gain: the envelope
/// sampled at the (clamped) playhead plus the keyframe row state driving the
/// volume row's diamond. The audio analogue of [`sample_transform`].
pub fn sample_audio(clip: &Clip, playhead: i32) -> AudioSample {
    AudioSample {
        volume: sampled_volume(clip, playhead),
        volume_row: row_state(&clip.kf_volume, playhead),
    }
}

/// Whether a "duck under voice" gesture makes sense for a clip on `track_id`:
/// true when some *other* audio lane is tagged as a voice source (the track
/// header "V" toggle, M8 Phase 4). Pure gate for the inspector button — the
/// worker re-resolves the precise overlapping voice clips when it fires.
pub fn can_duck_under_voice(sequence: Sequence, track_id: &str) -> bool {
    (0..sequence.tracks.row_count())
        .filter_map(|i| sequence.tracks.row_data(i))
        .any(|track| track.kind == TrackKind::Audio && track.duck_source && track.id != track_id)
}

pub fn resolve_selection(sequence: Sequence, track_id: &str, clip_id: &str) -> SelectedClipInfo {
    if track_id.is_empty() || clip_id.is_empty() {
        return SelectedClipInfo {
            found: false,
            track_kind: TrackKind::Video,
            clip: Clip::default(),
        };
    }

    for track_idx in 0..sequence.tracks.row_count() {
        let Some(track) = sequence.tracks.row_data(track_idx) else {
            continue;
        };
        if track.id != track_id {
            continue;
        }

        for clip_idx in 0..track.clips.row_count() {
            let Some(clip) = track.clips.row_data(clip_idx) else {
                continue;
            };
            if clip.id == clip_id {
                return SelectedClipInfo {
                    found: true,
                    track_kind: track.kind,
                    clip,
                };
            }
        }
    }

    SelectedClipInfo {
        found: false,
        track_kind: TrackKind::Video,
        clip: Clip::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Track;

    use slint::{ModelRc, SharedString, VecModel};
    use std::rc::Rc;

    fn track(id: &str, kind: TrackKind, duck_source: bool) -> Track {
        Track {
            id: SharedString::from(id),
            name: SharedString::from(id),
            kind,
            color: slint::Color::default(),
            clips: ModelRc::default(),
            enabled: true,
            muted: false,
            locked: false,
            duck_source,
            pinned: false,
            is_main: false,
            transitions: ModelRc::default(),
        }
    }

    fn sequence(tracks: Vec<Track>) -> Sequence {
        Sequence {
            tracks: ModelRc::from(Rc::new(VecModel::from(tracks))),
            ..Default::default()
        }
    }

    #[test]
    fn compensate_anchor_preserves_composited_center() {
        use crate::preview_select::{canvas_config, clip_placement};
        use crate::{Rational, RationalTime, TimeRange};

        let clip = Clip {
            media_id: SharedString::from("m1"),
            media_width: 1920,
            media_height: 1080,
            transform_scale: 1.0,
            transform_opacity: 1.0,
            transform_anchor_x: 0.5,
            transform_anchor_y: 0.5,
            timeline_start: RationalTime {
                value: 0,
                rate: Rational { num: 24, den: 1 },
            },
            source_range: TimeRange {
                start: RationalTime {
                    value: 0,
                    rate: Rational { num: 24, den: 1 },
                },
                duration: RationalTime {
                    value: 100,
                    rate: Rational { num: 24, den: 1 },
                },
            },
            ..Default::default()
        };
        let sequence = Sequence {
            width: 1920.0,
            height: 1080.0,
            ..Default::default()
        };
        let canvas = canvas_config(&sequence);
        let before = clip_placement(&clip, &canvas).center;
        let c = compensate_anchor_position(&clip, sequence, 10, 0.2, 0.8, 1.0, 0.0);
        let mut after_clip = clip.clone();
        after_clip.transform_position_x = c.position_x;
        after_clip.transform_position_y = c.position_y;
        after_clip.transform_anchor_x = 0.2;
        after_clip.transform_anchor_y = 0.8;
        let after = clip_placement(&after_clip, &canvas).center;
        assert!((after[0] - before[0]).abs() < 1e-2);
        assert!((after[1] - before[1]).abs() < 1e-2);
    }

    #[test]
    fn duck_gate_needs_a_voice_lane_other_than_the_clips_own() {
        // Lane "1" is plain music, lane "2" is tagged as the voice source.
        let seq = sequence(vec![
            track("1", TrackKind::Audio, false),
            track("2", TrackKind::Audio, true),
        ]);
        // A music clip on "1" can duck under the voice on "2".
        assert!(can_duck_under_voice(seq.clone(), "1"));
        // From the voice lane itself there is no *other* voice lane.
        assert!(!can_duck_under_voice(seq, "2"));
    }

    #[test]
    fn duck_gate_is_false_without_any_voice_lane() {
        let seq = sequence(vec![
            track("1", TrackKind::Audio, false),
            track("2", TrackKind::Audio, false),
        ]);
        assert!(!can_duck_under_voice(seq, "1"));
    }

    #[test]
    fn duck_gate_ignores_a_voice_flag_on_a_non_audio_lane() {
        // A duck_source flag is inert on a video lane (the toggle is audio-only).
        let seq = sequence(vec![
            track("1", TrackKind::Audio, false),
            track("2", TrackKind::Video, true),
        ]);
        assert!(!can_duck_under_voice(seq, "1"));
    }
}
