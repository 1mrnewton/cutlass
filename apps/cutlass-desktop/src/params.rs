//! UI-side helpers for projected keyframe curves (keyframes roadmap Phase 1).
//!
//! The projection publishes each animatable clip property as its clip-start
//! sample (`transform-*`) plus the keyframe list (`kf-*`, absolute sequence
//! ticks). These helpers rebuild a [`cutlass_models::Param`] from that data
//! and sample it with the engine's own math, so inspector value rows and
//! preview selection geometry can be playhead-accurate without a projection
//! republish per tick.

use cutlass_models::{ClipTransform, Easing, Keyframe, Param};
use slint::Model;

use crate::{Clip, ParamKeyframe, ParamRowState};

/// Encode an engine easing for the Slint `ParamKeyframe`:
/// `(tag, [x1, y1, x2, y2])` — points are zero for the built-in presets.
///
/// Tags: 0 linear, 1 ease-in, 2 ease-out, 3 ease-in-out, 4 custom bezier,
/// 5 snappy, 6 overshoot, 7 anticipate ([`cutlass_models::EASING_PRESETS`]),
/// 8 hold.
pub(crate) fn easing_to_ui(easing: Easing) -> (i32, [f32; 4]) {
    match easing {
        Easing::Linear => (0, [0.0; 4]),
        Easing::EaseIn => (1, [0.0; 4]),
        Easing::EaseOut => (2, [0.0; 4]),
        Easing::EaseInOut => (3, [0.0; 4]),
        Easing::Hold => (8, [0.0; 4]),
        Easing::Bezier { points } => match easing.preset_id() {
            Some("snappy") => (5, points),
            Some("overshoot") => (6, points),
            Some("anticipate") => (7, points),
            _ => (4, points),
        },
    }
}

/// Decode the Slint easing encoding back to the engine enum. Unknown tags
/// fall back to linear.
pub(crate) fn easing_from_ui(tag: i32, points: [f32; 4]) -> Easing {
    match tag {
        1 => Easing::EaseIn,
        2 => Easing::EaseOut,
        3 => Easing::EaseInOut,
        4 => Easing::Bezier { points },
        5 => Easing::from_preset_id("snappy").unwrap_or(Easing::Linear),
        6 => Easing::from_preset_id("overshoot").unwrap_or(Easing::Linear),
        7 => Easing::from_preset_id("anticipate").unwrap_or(Easing::Linear),
        8 => Easing::Hold,
        _ => Easing::Linear,
    }
}

fn easing_of(kf: &ParamKeyframe) -> Easing {
    easing_from_ui(kf.easing, [kf.bez_x1, kf.bez_y1, kf.bez_x2, kf.bez_y2])
}

/// Rebuild a scalar `Param` from a projected keyframe list (absolute ticks).
/// An empty list ⇔ the constant published in the `transform-*` field.
fn scalar_param(kfs: &slint::ModelRc<ParamKeyframe>, constant: f32) -> Param<f32> {
    let keyframes: Vec<Keyframe<f32>> = kfs
        .iter()
        .map(|kf| Keyframe {
            tick: i64::from(kf.tick),
            value: kf.value_x,
            easing: easing_of(&kf),
        })
        .collect();
    if keyframes.is_empty() {
        Param::Constant(constant)
    } else {
        Param::Keyframed { keyframes }
    }
}

fn vec2_param(kfs: &slint::ModelRc<ParamKeyframe>, constant: [f32; 2]) -> Param<[f32; 2]> {
    let keyframes: Vec<Keyframe<[f32; 2]>> = kfs
        .iter()
        .map(|kf| Keyframe {
            tick: i64::from(kf.tick),
            value: [kf.value_x, kf.value_y],
            easing: easing_of(&kf),
        })
        .collect();
    if keyframes.is_empty() {
        Param::Constant(constant)
    } else {
        Param::Keyframed { keyframes }
    }
}

/// Clamp the playhead into the clip's extent, mirroring the engine's
/// `Clip::animation_tick` (a clip's animation holds its first/last frame
/// value outside the clip).
fn clamped_tick(clip: &Clip, playhead: i32) -> i64 {
    let start = i64::from(clip.timeline_start.value);
    let last = start + i64::from(clip.source_range.duration.value.max(1)) - 1;
    i64::from(playhead).clamp(start, last)
}

/// The clip's transform sampled at the playhead — identical math to the
/// engine's `resolve_layers` sample for the composited frame.
pub(crate) fn sampled_transform(clip: &Clip, playhead: i32) -> ClipTransform {
    let tick = clamped_tick(clip, playhead);
    ClipTransform {
        position: vec2_param(
            &clip.kf_position,
            [clip.transform_position_x, clip.transform_position_y],
        )
        .sample(tick),
        anchor_point: vec2_param(
            &clip.kf_anchor,
            [clip.transform_anchor_x, clip.transform_anchor_y],
        )
        .sample(tick),
        scale: scalar_param(&clip.kf_scale, clip.transform_scale).sample(tick),
        rotation: scalar_param(&clip.kf_rotation, clip.transform_rotation).sample(tick),
        opacity: scalar_param(&clip.kf_opacity, clip.transform_opacity).sample(tick),
    }
}

/// The clip's audio gain sampled at the (clamped) playhead — the same
/// `Param` math the mixers use, so the inspector readout and diamond track
/// exactly what's heard. An empty `kf-volume` ⇔ the constant in `volume`.
pub(crate) fn sampled_volume(clip: &Clip, playhead: i32) -> f32 {
    let tick = clamped_tick(clip, playhead);
    scalar_param(&clip.kf_volume, clip.volume).sample(tick)
}

/// Sample one scalar text-style or color-look property at the playhead. The
/// string keys deliberately match the inspector command keys, keeping each
/// property's display value and diamond state on the same projected curve.
pub(crate) fn sampled_scalar_param(clip: &Clip, param: &str, playhead: i32) -> Option<f32> {
    let tick = clamped_tick(clip, playhead);
    let (keyframes, constant) = match param {
        "text_size" => (&clip.kf_text_size, clip.text_style.size),
        "text_letter_spacing" => (&clip.kf_text_letter_spacing, clip.text_style.letter_spacing),
        "text_line_spacing" => (&clip.kf_text_line_spacing, clip.text_style.line_spacing),
        "text_stroke_width" => (&clip.kf_text_stroke_width, clip.text_style.stroke_width),
        "text_shadow_blur" => (&clip.kf_text_shadow_blur, clip.text_style.shadow_blur),
        "text_shadow_distance" => (
            &clip.kf_text_shadow_distance,
            clip.text_style.shadow_distance,
        ),
        "look_filter_intensity" => (&clip.kf_look_filter_intensity, clip.filter_intensity),
        "look_lut_intensity" => (&clip.kf_look_lut_intensity, clip.lut_intensity),
        "look_adjust_brightness" => (&clip.kf_look_adjust_brightness, clip.adjust_brightness),
        "look_adjust_contrast" => (&clip.kf_look_adjust_contrast, clip.adjust_contrast),
        "look_adjust_saturation" => (&clip.kf_look_adjust_saturation, clip.adjust_saturation),
        "look_adjust_exposure" => (&clip.kf_look_adjust_exposure, clip.adjust_exposure),
        "look_adjust_temperature" => (&clip.kf_look_adjust_temperature, clip.adjust_temperature),
        "style_shadow_blur" => (&clip.kf_style_shadow_blur, clip.style_shadow_blur),
        "style_glow_radius" => (&clip.kf_style_glow_radius, clip.style_glow_radius),
        "style_glow_intensity" => (&clip.kf_style_glow_intensity, clip.style_glow_intensity),
        "style_outline_width" => (&clip.kf_style_outline_width, clip.style_outline_width),
        "style_background_padding" => (
            &clip.kf_style_background_padding,
            clip.style_background_padding,
        ),
        "style_background_radius" => (
            &clip.kf_style_background_radius,
            clip.style_background_radius,
        ),
        "look_mask_feather" => (&clip.kf_look_mask_feather, clip.mask_feather),
        "look_mask_rotation" => (&clip.kf_look_mask_rotation, clip.mask_rotation),
        "look_mask_roundness" => (&clip.kf_look_mask_roundness, clip.mask_roundness),
        _ => return None,
    };
    Some(scalar_param(keyframes, constant).sample(tick))
}

/// Sample a vec2 param (`position`, `anchor`, `style_shadow_offset`,
/// `look_mask_center`, `look_mask_size`) at the playhead. Axis-display keys
/// (`*_x` / `*_y`) are handled in [`crate::inspector::sample_scalar_param`].
pub(crate) fn sampled_vec2_param(clip: &Clip, param: &str, playhead: i32) -> Option<[f32; 2]> {
    let tick = clamped_tick(clip, playhead);
    let (keyframes, constant) = match param {
        "position" => (
            &clip.kf_position,
            [clip.transform_position_x, clip.transform_position_y],
        ),
        "anchor" => (
            &clip.kf_anchor,
            [clip.transform_anchor_x, clip.transform_anchor_y],
        ),
        "style_shadow_offset" => (
            &clip.kf_style_shadow_offset,
            [clip.style_shadow_offset_x, clip.style_shadow_offset_y],
        ),
        "look_mask_center" => (
            &clip.kf_look_mask_center,
            [clip.mask_center_x, clip.mask_center_y],
        ),
        "look_mask_size" => (
            &clip.kf_look_mask_size,
            [clip.mask_size_w, clip.mask_size_h],
        ),
        _ => return None,
    };
    Some(vec2_param(keyframes, constant).sample(tick))
}

/// Overwrite the clip's `transform-*` fields with the playhead sample, so
/// geometry code that reads those fields (placement, hit-test, gestures)
/// follows the rendered frame on animated clips.
pub(crate) fn apply_sampled_transform(clip: &mut Clip, playhead: i32) {
    let t = sampled_transform(clip, playhead);
    clip.transform_position_x = t.position[0];
    clip.transform_position_y = t.position[1];
    clip.transform_anchor_x = t.anchor_point[0];
    clip.transform_anchor_y = t.anchor_point[1];
    clip.transform_scale = t.scale;
    clip.transform_rotation = t.rotation;
    clip.transform_opacity = t.opacity;
}

/// Merged, deduped keyframe ticks (absolute, ascending) across every
/// animated property — the timeline draws one diamond per tick on the
/// selected clip (keyframes roadmap Phase 2), CapCut-style.
pub(crate) fn merged_keyframe_ticks(clip: &Clip) -> slint::ModelRc<i32> {
    let mut ticks: Vec<i32> = [
        &clip.kf_position,
        &clip.kf_anchor,
        &clip.kf_scale,
        &clip.kf_rotation,
        &clip.kf_opacity,
        &clip.kf_text_size,
        &clip.kf_text_fill,
        &clip.kf_text_letter_spacing,
        &clip.kf_text_line_spacing,
        &clip.kf_text_stroke_width,
        &clip.kf_text_stroke_color,
        &clip.kf_text_shadow_blur,
        &clip.kf_text_shadow_distance,
        &clip.kf_text_shadow_color,
        &clip.kf_look_filter_intensity,
        &clip.kf_look_lut_intensity,
        &clip.kf_look_adjust_brightness,
        &clip.kf_look_adjust_contrast,
        &clip.kf_look_adjust_saturation,
        &clip.kf_look_adjust_exposure,
        &clip.kf_look_adjust_temperature,
        &clip.kf_style_shadow_offset,
        &clip.kf_style_shadow_blur,
        &clip.kf_style_glow_radius,
        &clip.kf_style_glow_intensity,
        &clip.kf_style_outline_width,
        &clip.kf_style_background_padding,
        &clip.kf_style_background_radius,
        &clip.kf_look_mask_feather,
        &clip.kf_look_mask_center,
        &clip.kf_look_mask_size,
        &clip.kf_look_mask_rotation,
        &clip.kf_look_mask_roundness,
    ]
    .iter()
    .flat_map(|kfs| kfs.iter().map(|kf| kf.tick))
    .collect();
    ticks.sort_unstable();
    ticks.dedup();
    slint::ModelRc::from(std::rc::Rc::new(slint::VecModel::from(ticks)))
}

/// Keyframe row state for one property at the playhead: drives the
/// inspector's diamond (add/remove at playhead) and ◀ ▶ navigation.
pub(crate) fn row_state(kfs: &slint::ModelRc<ParamKeyframe>, playhead: i32) -> ParamRowState {
    let mut state = ParamRowState {
        animated: false,
        on_keyframe: false,
        prev_tick: -1,
        next_tick: -1,
        easing: 0,
    };
    for kf in kfs.iter() {
        state.animated = true;
        if kf.tick == playhead {
            state.on_keyframe = true;
            state.easing = kf.easing;
        } else if kf.tick < playhead {
            // Ticks arrive sorted; the last one below the playhead wins.
            state.prev_tick = kf.tick;
        } else if state.next_tick < 0 {
            state.next_tick = kf.tick;
        }
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Rational, RationalTime, TimeRange};
    use slint::{ModelRc, VecModel};
    use std::rc::Rc;

    fn kf(tick: i32, value: f32) -> ParamKeyframe {
        ParamKeyframe {
            tick,
            value_x: value,
            value_y: 0.0,
            easing: 0,
            bez_x1: 0.0,
            bez_y1: 0.0,
            bez_x2: 0.0,
            bez_y2: 0.0,
        }
    }

    fn kfs(items: Vec<ParamKeyframe>) -> ModelRc<ParamKeyframe> {
        ModelRc::from(Rc::new(VecModel::from(items)))
    }

    fn rt(value: i32) -> RationalTime {
        RationalTime {
            value,
            rate: Rational { num: 24, den: 1 },
        }
    }

    /// Clip [start, start+dur) with constant identity transform.
    fn clip(start: i32, dur: i32) -> Clip {
        Clip {
            timeline_start: rt(start),
            source_range: TimeRange {
                start: rt(0),
                duration: rt(dur),
            },
            transform_scale: 1.0,
            transform_opacity: 1.0,
            transform_anchor_x: 0.5,
            transform_anchor_y: 0.5,
            ..Default::default()
        }
    }

    #[test]
    fn constant_clip_samples_published_fields() {
        let mut c = clip(0, 100);
        c.transform_position_x = 0.25;
        c.transform_rotation = 45.0;
        let t = sampled_transform(&c, 50);
        assert_eq!(t.position, [0.25, 0.0]);
        assert_eq!(t.rotation, 45.0);
        assert_eq!(t.scale, 1.0);
    }

    #[test]
    fn animated_scale_samples_at_playhead_and_clamps() {
        let mut c = clip(100, 50);
        c.kf_scale = kfs(vec![kf(100, 1.0), kf(140, 2.0)]);
        assert_eq!(sampled_transform(&c, 120).scale, 1.5);
        assert_eq!(sampled_transform(&c, 100).scale, 1.0);
        // Before the clip / after the last keyframe: first / last value holds.
        assert_eq!(sampled_transform(&c, 0).scale, 1.0);
        assert_eq!(sampled_transform(&c, 1000).scale, 2.0);
    }

    #[test]
    fn eased_keyframe_matches_engine_curve() {
        let mut c = clip(0, 100);
        let mut eased = kf(0, 0.0);
        eased.easing = 1; // ease-in (quadratic): halfway = 0.25
        c.kf_opacity = kfs(vec![eased, kf(40, 1.0)]);
        let t = sampled_transform(&c, 20);
        assert!((t.opacity - 0.25).abs() < 1e-6, "got {}", t.opacity);
    }

    #[test]
    fn position_keyframes_carry_both_axes() {
        let mut c = clip(0, 100);
        let mut a = kf(0, -0.5);
        a.value_y = 0.0;
        let mut b = kf(10, 0.5);
        b.value_y = 1.0;
        c.kf_position = kfs(vec![a, b]);
        let t = sampled_transform(&c, 5);
        assert_eq!(t.position, [0.0, 0.5]);
    }

    #[test]
    fn row_state_reports_neighbors_and_hit() {
        let curve = kfs(vec![kf(10, 0.0), kf(20, 1.0), kf(30, 2.0)]);
        let s = row_state(&curve, 20);
        assert!(s.animated && s.on_keyframe);
        assert_eq!((s.prev_tick, s.next_tick), (10, 30));

        let s = row_state(&curve, 25);
        assert!(s.animated && !s.on_keyframe);
        assert_eq!((s.prev_tick, s.next_tick), (20, 30));

        let s = row_state(&curve, 5);
        assert_eq!((s.prev_tick, s.next_tick), (-1, 10));
        let s = row_state(&curve, 35);
        assert_eq!((s.prev_tick, s.next_tick), (30, -1));

        let s = row_state(&kfs(vec![]), 5);
        assert!(!s.animated && !s.on_keyframe);
        assert_eq!((s.prev_tick, s.next_tick), (-1, -1));
    }

    #[test]
    fn row_state_easing_reflects_keyframe_under_playhead() {
        let mut eased = kf(10, 0.0);
        eased.easing = 3;
        let curve = kfs(vec![eased, kf(20, 1.0)]);
        assert_eq!(row_state(&curve, 10).easing, 3);
        assert_eq!(row_state(&curve, 20).easing, 0);
    }

    #[test]
    fn merged_ticks_dedup_across_properties_in_order() {
        let mut c = clip(0, 100);
        c.kf_scale = kfs(vec![kf(10, 1.0), kf(30, 2.0)]);
        c.kf_opacity = kfs(vec![kf(5, 0.0), kf(30, 1.0)]);
        let mut pos = kf(30, 0.1);
        pos.value_y = 0.2;
        c.kf_position = kfs(vec![pos]);

        let merged = merged_keyframe_ticks(&c);
        let ticks: Vec<i32> = merged.iter().collect();
        assert_eq!(ticks, vec![5, 10, 30]);

        assert_eq!(merged_keyframe_ticks(&clip(0, 10)).row_count(), 0);
    }

    #[test]
    fn style_shadow_blur_samples_projected_curve() {
        let mut c = clip(0, 100);
        c.style_shadow_blur = 8.0;
        assert_eq!(sampled_scalar_param(&c, "style_shadow_blur", 50), Some(8.0));
        c.kf_style_shadow_blur = kfs(vec![kf(0, 4.0), kf(40, 12.0)]);
        assert_eq!(sampled_scalar_param(&c, "style_shadow_blur", 20), Some(8.0));
        assert_eq!(
            sampled_vec2_param(&c, "style_shadow_offset", 0),
            Some([0.0, 0.0])
        );
        c.style_shadow_offset_x = 4.0;
        c.style_shadow_offset_y = -2.0;
        assert_eq!(
            sampled_vec2_param(&c, "style_shadow_offset", 10),
            Some([4.0, -2.0])
        );
    }

    #[test]
    fn easing_roundtrips_through_ui_encoding() {
        for easing in [
            Easing::Linear,
            Easing::EaseIn,
            Easing::EaseOut,
            Easing::EaseInOut,
            Easing::from_preset_id("snappy").unwrap(),
            Easing::from_preset_id("overshoot").unwrap(),
            Easing::from_preset_id("anticipate").unwrap(),
            Easing::Bezier {
                points: [0.42, 0.0, 0.58, 1.0],
            },
            Easing::Hold,
        ] {
            let (tag, points) = easing_to_ui(easing);
            assert_eq!(easing_from_ui(tag, points), easing);
        }
    }
}
