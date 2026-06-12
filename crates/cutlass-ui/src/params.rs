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
/// `(tag, [x1, y1, x2, y2])` — points are zero for the presets.
pub(crate) fn easing_to_ui(easing: Easing) -> (i32, [f32; 4]) {
    match easing {
        Easing::Linear => (0, [0.0; 4]),
        Easing::EaseIn => (1, [0.0; 4]),
        Easing::EaseOut => (2, [0.0; 4]),
        Easing::EaseInOut => (3, [0.0; 4]),
        Easing::Bezier { points } => (4, points),
    }
}

/// Decode the Slint easing encoding back to the engine enum. Unknown tags
/// fall back to linear (defensive: the UI only emits 0..=3 today).
pub(crate) fn easing_from_ui(tag: i32, points: [f32; 4]) -> Easing {
    match tag {
        1 => Easing::EaseIn,
        2 => Easing::EaseOut,
        3 => Easing::EaseInOut,
        4 => Easing::Bezier { points },
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
        scale: scalar_param(&clip.kf_scale, clip.transform_scale).sample(tick),
        rotation: scalar_param(&clip.kf_rotation, clip.transform_rotation).sample(tick),
        opacity: scalar_param(&clip.kf_opacity, clip.transform_opacity).sample(tick),
    }
}

/// Overwrite the clip's `transform-*` fields with the playhead sample, so
/// geometry code that reads those fields (placement, hit-test, gestures)
/// follows the rendered frame on animated clips.
pub(crate) fn apply_sampled_transform(clip: &mut Clip, playhead: i32) {
    let t = sampled_transform(clip, playhead);
    clip.transform_position_x = t.position[0];
    clip.transform_position_y = t.position[1];
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
        &clip.kf_scale,
        &clip.kf_rotation,
        &clip.kf_opacity,
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
    fn easing_roundtrips_through_ui_encoding() {
        for easing in [
            Easing::Linear,
            Easing::EaseIn,
            Easing::EaseOut,
            Easing::EaseInOut,
            Easing::Bezier {
                points: [0.42, 0.0, 0.58, 1.0],
            },
        ] {
            let (tag, points) = easing_to_ui(easing);
            assert_eq!(easing_from_ui(tag, points), easing);
        }
    }
}
