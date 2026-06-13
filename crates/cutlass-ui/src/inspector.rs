//! Inspector helpers: resolve the selected clip for the property sheet, and
//! sample its animated transform at the playhead for the keyframe UI.

use crate::params::{row_state, sampled_transform, sampled_volume};
use crate::{
    AudioSample, Clip, SelectedClipInfo, Sequence, TextClipStyle, TrackKind, TransformSample,
};
use cutlass_models::{
    TextAlignH, TextAlignV, TextBackground, TextCase, TextShadow, TextStroke,
    TextStyle as ModelTextStyle,
};
use slint::Model;

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
        size: style.size,
        bold: style.bold,
        italic: style.italic,
        underline: style.underline,
        case: text_case_from_int(style.case),
        fill: rgba(style.fill),
        letter_spacing: style.letter_spacing,
        line_spacing: style.line_spacing,
        align_h: align_h_from_int(style.align_h),
        align_v: align_v_from_int(style.align_v),
        wrap: style.wrap,
        stroke: style.stroke_enabled.then(|| TextStroke {
            rgba: rgba(style.stroke_color),
            width: style.stroke_width,
        }),
        background: style.background_enabled.then(|| TextBackground {
            rgba: rgb_alpha(style.background_color, style.background_opacity),
            radius: style.background_radius,
        }),
        shadow: style.shadow_enabled.then(|| TextShadow {
            rgba: rgb_alpha(style.shadow_color, style.shadow_opacity),
            blur: style.shadow_blur,
            distance: style.shadow_distance,
        }),
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
        scale: t.scale,
        rotation: t.rotation,
        opacity: t.opacity,
        position_row: row_state(&clip.kf_position, playhead),
        scale_row: row_state(&clip.kf_scale, playhead),
        rotation_row: row_state(&clip.kf_rotation, playhead),
        opacity_row: row_state(&clip.kf_opacity, playhead),
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
        .any(|track| {
            track.kind == TrackKind::Audio && track.duck_source && track.id != track_id
        })
}

pub fn resolve_selection(
    sequence: Sequence,
    track_id: &str,
    clip_id: &str,
) -> SelectedClipInfo {
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
